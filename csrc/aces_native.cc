// PROTOTYPE (not ready to merge): native pass for ACES's MEDS plain predicates (aces.predicates.generate_plain_predicates_from_meds):
// one pass over a shard's Arrow record batches, without Python, so shards can run on threads concurrently.
//
// Python evaluates each predicate's code condition on every batch's code dictionary (with ACES's own Polars
// expression, so regex/any/exact semantics are ACES's) and passes one flag row per dictionary entry. Here:
//   1. rows with a null or MEDS_BIRTH code are dropped (ACES's `code != birth_code` filter);
//   2. with prune, each subject's first and last timestamps are found (over the remaining rows);
//   3. each predicate is its code flag and its value bounds (Polars semantics: NaN above every number, a null
//      value makes the predicate null, counted as 0);
//   4. with prune, a row is kept if its timestamp is null, its subject's first or last, or a predicate holds;
//   5. kept rows are summed per (subject, timestamp), in order of first appearance.
#include <cmath>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#include <cstdint>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

extern "C" {

typedef struct {
  const uint8_t* validity;  // Arrow validity bitmap, or NULL when the column has no nulls.
  const void* data;
  int64_t offset;  // The array's offset, in elements.
} acn_column;

typedef struct {
  int64_t length;
  acn_column subjects;  // int64
  acn_column times;     // int64 (timestamp); null is a static record
  acn_column codes;     // int32 indices into this batch's code dictionary
  // dictionary_length x (predicates + 1) bytes: column 0 is 1 when the entry is dropped (MEDS_BIRTH or a null
  // entry); column 1 + p is 1 when the entry satisfies predicate p's code condition.
  const uint8_t* code_flags;
  int64_t dictionary_length;
  acn_column values;  // float32 (values_64 = 0) or float64; may have NULL data when no bounds are used
  int32_t values_64;
} acn_batch;

typedef struct {
  int32_t has_min, min_inclusive, has_max, max_inclusive;
  double min, max;  // Already rounded to the value column's type.
} acn_bounds;

typedef struct acn_result acn_result;

}  // extern "C"

namespace {

thread_local std::string last_error;

inline bool Valid(const acn_column& c, int64_t i) {
  if (!c.validity) return true;
  const int64_t bit = c.offset + i;
  return (c.validity[bit >> 3] >> (bit & 7)) & 1;
}

template <typename T>
inline const T* Data(const acn_column& c) {
  return static_cast<const T*>(c.data) + c.offset;
}

// Polars' float comparisons: NaN is above every number (and equal to NaN).
inline bool Ge(double a, double b) { return std::isnan(a) ? true : (std::isnan(b) ? false : a >= b); }
inline int Ctz(uint32_t x) {
#if defined(_MSC_VER)
  unsigned long i;
  _BitScanForward(&i, x);
  return static_cast<int>(i);
#else
  return __builtin_ctz(x);
#endif
}

inline bool Gt(double a, double b) { return std::isnan(a) ? !std::isnan(b) : (std::isnan(b) ? false : a > b); }

struct Key {
  int64_t subject;
  int64_t time;
  uint8_t subject_null, time_null;
  bool operator==(const Key& o) const {
    return subject_null == o.subject_null && time_null == o.time_null &&
           (subject_null || subject == o.subject) && (time_null || time == o.time);
  }
};

struct KeyHash {
  size_t operator()(const Key& k) const {
    uint64_t h = static_cast<uint64_t>(k.subject_null ? 0x9E3779B97F4A7C15ull : k.subject) * 0xBF58476D1CE4E5B9ull;
    h ^= static_cast<uint64_t>(k.time_null ? 0x94D049BB133111EBull : k.time) + (h << 6) + (h >> 2);
    h *= 0x94D049BB133111EBull;
    return static_cast<size_t>(h ^ (h >> 31));
  }
};

struct SubjectKey {
  int64_t subject;
  uint8_t null;
  bool operator==(const SubjectKey& o) const { return null == o.null && (null || subject == o.subject); }
};

struct SubjectHash {
  size_t operator()(const SubjectKey& k) const {
    uint64_t h = static_cast<uint64_t>(k.null ? 0x9E3779B97F4A7C15ull : k.subject) * 0xBF58476D1CE4E5B9ull;
    return static_cast<size_t>(h ^ (h >> 29));
  }
};

struct Bounds {  // A subject's first and last non-null timestamps.
  int64_t first = 0, last = 0;
  bool any = false;
};

}  // namespace

struct acn_result {
  std::vector<int64_t> subjects, times, counts;
  std::vector<uint8_t> subject_valid, time_valid;
};

namespace {

template <typename Value>
void Run(const acn_batch* batches, int64_t count, int32_t predicates, const acn_bounds* bounds, bool prune,
         acn_result& out) {
  if (predicates > 31) throw std::invalid_argument("At most 31 predicates.");
  const int32_t width = predicates + 1;
  constexpr uint32_t kDrop = 1u << 31;
  uint32_t bounded = 0;  // Predicates with value bounds.
  for (int32_t p = 0; p < predicates; ++p) {
    if (bounds[p].has_min || bounds[p].has_max) bounded |= 1u << p;
  }
  // Each batch's dictionary entries as one mask: kDrop (MEDS_BIRTH or a null code), and bit p when the entry
  // satisfies predicate p's code condition.
  std::vector<std::vector<uint32_t>> masks(static_cast<size_t>(count));
  for (int64_t b = 0; b < count; ++b) {
    const acn_batch& r = batches[b];
    if (b > 0 && r.code_flags == batches[b - 1].code_flags && r.dictionary_length == batches[b - 1].dictionary_length) {
      masks[b] = masks[b - 1];
      continue;
    }
    auto& m = masks[b];
    m.assign(static_cast<size_t>(r.dictionary_length), 0);
    for (int64_t d = 0; d < r.dictionary_length; ++d) {
      const uint8_t* flags = r.code_flags + d * width;
      uint32_t bits = flags[0] ? kDrop : 0;
      for (int32_t p = 0; p < predicates; ++p) bits |= static_cast<uint32_t>(flags[1 + p] != 0) << p;
      m[static_cast<size_t>(d)] = bits;
    }
  }
  const auto code_mask = [&](const acn_batch& r, const std::vector<uint32_t>& m, int64_t i) -> uint32_t {
    if (r.codes.validity && !Valid(r.codes, i)) return kDrop;  // A null code: dropped by `code != birth`.
    const int32_t code = Data<int32_t>(r.codes)[i];
    if (code < 0 || code >= r.dictionary_length) throw std::out_of_range("Code index out of the dictionary.");
    return m[static_cast<size_t>(code)];
  };

  // 1-2. Each subject's first and last timestamps over the rows that survive the code filter.
  std::unordered_map<SubjectKey, Bounds, SubjectHash> spans;
  if (prune) {
    for (int64_t b = 0; b < count; ++b) {
      const acn_batch& r = batches[b];
      const auto& m = masks[b];
      const auto* subjects = Data<int64_t>(r.subjects);
      const auto* times = Data<int64_t>(r.times);
      Bounds* span = nullptr;
      SubjectKey last{0, 2};
      for (int64_t i = 0; i < r.length; ++i) {
        if (code_mask(r, m, i) & kDrop) continue;
        const bool subject_valid = !r.subjects.validity || Valid(r.subjects, i);
        const SubjectKey key{subject_valid ? subjects[i] : 0, static_cast<uint8_t>(!subject_valid)};
        if (!span || !(key == last)) {
          span = &spans[key];
          last = key;
        }
        if (r.times.validity && !Valid(r.times, i)) continue;
        const int64_t t = times[i];
        if (!span->any) {
          span->first = span->last = t;
          span->any = true;
        } else {
          if (t < span->first) span->first = t;
          if (t > span->last) span->last = t;
        }
      }
    }
  }

  // 3-5. Predicates, the prune filter, and the sums per (subject, timestamp) in first-appearance order.
  // MEDS shards are sorted by subject, then time (static rows first): while each new key is greater than the
  // newest row's, it cannot have appeared before, so rows are appended without a lookup (the newest row's key
  // again, e.g. across record batches, is that row). The first key out of order switches to a map of every key
  // so far, which keeps first-appearance order for any input.
  std::unordered_map<Key, int64_t, KeyHash> rows;
  bool ordered = true;
  const auto before = [](const Key& a, const Key& b) {  // a < b in (subject: null first, time: null first)
    if (a.subject_null != b.subject_null) return a.subject_null > b.subject_null;
    if (!a.subject_null && a.subject != b.subject) return a.subject < b.subject;
    if (a.time_null != b.time_null) return a.time_null > b.time_null;
    return !a.time_null && a.time < b.time;
  };
  Key last_key{0, 0, 2, 2};
  int64_t last_row = -1;
  const Bounds* span = nullptr;
  SubjectKey span_key{0, 2};
  for (int64_t b = 0; b < count; ++b) {
    const acn_batch& r = batches[b];
    const auto& m = masks[b];
    const auto* subjects = Data<int64_t>(r.subjects);
    const auto* times = Data<int64_t>(r.times);
    const Value* values = r.values.data ? Data<Value>(r.values) : nullptr;
    for (int64_t i = 0; i < r.length; ++i) {
      uint32_t bits = code_mask(r, m, i);
      if (bits & kDrop) continue;
      if (bits & bounded) {
        const bool has_value = values && (!r.values.validity || Valid(r.values, i));
        const double v = has_value ? static_cast<double>(values[i]) : 0.0;
        for (uint32_t check = bits & bounded; check; check &= check - 1) {
          const int p = Ctz(check);
          const acn_bounds& bd = bounds[p];
          bool hit = has_value;  // A null value makes the predicate null: counted as 0.
          if (hit && bd.has_min) hit = bd.min_inclusive ? Ge(v, bd.min) : Gt(v, bd.min);
          if (hit && bd.has_max) hit = bd.max_inclusive ? Ge(bd.max, v) : Gt(bd.max, v);
          if (!hit) bits &= ~(1u << p);
        }
      }
      const bool subject_valid = !r.subjects.validity || Valid(r.subjects, i);
      const bool time_valid = !r.times.validity || Valid(r.times, i);
      const Key key{subject_valid ? subjects[i] : 0, time_valid ? times[i] : 0, static_cast<uint8_t>(!subject_valid),
                    static_cast<uint8_t>(!time_valid)};
      if (prune && time_valid && !bits) {
        const SubjectKey sk{key.subject, key.subject_null};
        if (!span || !(sk == span_key)) {
          span = &spans.at(sk);
          span_key = sk;
        }
        if (!(span->any && (key.time == span->first || key.time == span->last))) continue;
      }
      int64_t row;
      if (last_row >= 0 && key == last_key) {
        row = last_row;
      } else {
        const int64_t rows_so_far = static_cast<int64_t>(out.subjects.size());
        bool newest_again = false;
        if (ordered && rows_so_far > 0) {
          const Key newest{out.subjects.back(), out.times.back(), static_cast<uint8_t>(!out.subject_valid.back()),
                           static_cast<uint8_t>(!out.time_valid.back())};
          newest_again = newest == key;
          if (!newest_again && !before(newest, key)) {
            ordered = false;
            rows.reserve(static_cast<size_t>(rows_so_far) * 2);
            for (int64_t k = 0; k < rows_so_far; ++k) {
              rows.emplace(Key{out.subjects[k], out.times[k], static_cast<uint8_t>(!out.subject_valid[k]),
                               static_cast<uint8_t>(!out.time_valid[k])},
                           k);
            }
          }
        }
        bool fresh = true;
        if (ordered && newest_again) {
          row = rows_so_far - 1;
          fresh = false;
        } else if (ordered) {
          row = rows_so_far;
        } else {
          const auto found = rows.emplace(key, rows_so_far);
          row = found.first->second;
          fresh = found.second;
        }
        if (fresh) {
          out.subjects.push_back(key.subject);
          out.times.push_back(key.time);
          out.subject_valid.push_back(!key.subject_null);
          out.time_valid.push_back(!key.time_null);
          out.counts.resize(out.counts.size() + static_cast<size_t>(predicates), 0);
        }
        last_key = key;
        last_row = row;
      }
      int64_t* cell = out.counts.data() + row * predicates;
      for (; bits; bits &= bits - 1) ++cell[Ctz(bits)];
    }
  }
}

}  // namespace

extern "C" {

const char* acn_last_error(void) { return last_error.c_str(); }

acn_result* acn_meds_predicates(const acn_batch* batches, int64_t count, int32_t predicates,
                                const acn_bounds* bounds, int32_t prune, int32_t values_64) {
  last_error.clear();
  try {
    if (count < 0 || predicates < 0 || (count && !batches) || (predicates && !bounds)) {
      throw std::invalid_argument("Invalid arguments.");
    }
    auto* out = new acn_result();
    try {
      if (values_64) {
        Run<double>(batches, count, predicates, bounds, prune != 0, *out);
      } else {
        Run<float>(batches, count, predicates, bounds, prune != 0, *out);
      }
    } catch (...) {
      delete out;
      throw;
    }
    return out;
  } catch (const std::exception& e) {
    last_error = e.what();
  } catch (...) {
    last_error = "Unknown C++ exception.";
  }
  return nullptr;
}

int64_t acn_result_rows(const acn_result* r) { return static_cast<int64_t>(r->subjects.size()); }
const int64_t* acn_result_subjects(const acn_result* r) { return r->subjects.data(); }
const int64_t* acn_result_times(const acn_result* r) { return r->times.data(); }
const uint8_t* acn_result_subject_valid(const acn_result* r) { return r->subject_valid.data(); }
const uint8_t* acn_result_time_valid(const acn_result* r) { return r->time_valid.data(); }
const int64_t* acn_result_counts(const acn_result* r) { return r->counts.data(); }
void acn_result_free(acn_result* r) { delete r; }

}  // extern "C"
