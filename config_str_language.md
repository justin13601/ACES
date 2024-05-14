# Configuration Language Specification

## Introduction and Terminology

This document specifies the configuration language for the automatic extraction of task dataframes and cohorts
from structured EHR data organized either via the [MEDS](https://github.com/Medical-Event-Data-Standard/meds)
format (recommended) or the [ESGPT](https://eventstreamml.readthedocs.io/en/latest/) format. This extraction
system works by defining a configuration object that details the underlying concepts, inclusion/exclusion, and
labeling criteria for the cohort/task to be extracted, then using a recursive algorithm to identify all
realizations of valid patient time-ranges of data that satisfy those constraints from the raw data. For more
details on the recursive algorithm, see the `terminology.md` file. **TODO** better integrate, name, and link
to these documentation files.

As indicated above, these cohorts are specified through a combination of concepts (realized as event
_predicate_ functions, _aka_ "predicates") which are _dataset specific_ and inclusion/exclusion/labeling
criteria which, conditioned on a set of predicate definitions, are _dataset agnostic_.

Predicates are currently limited to "count" predicates, which are predicates that count the number of times a
boolean condition is satisfied over a given time window, which can either be a single timepoint, thus tracking
whether how many observations there were that satisfied the boolean condition in that event (_aka_ at that
timepoint) or over 1-dimensional windows. In the future, predicates may expand to include other notions of
functional characterization, such as tracking the average/min/max value a concept takes on over a time-period,
etc.

Constraints are specified in terms of time-points that can be bounded by events that satisfy predicates or
temporal relationships on said events. The windows between these time-points can then either be constrained to
contain events that satisfy certain aggregation functions over predicates for these time frames.

## Simplified Form (for human input)

TODO

## Machine Form (what is used by the algorithm)

In the machine form, the configuration file consists of two parts:

- `predicates`, stored as a dictionary from string predicate names (which must be unique) to either
  `DirectPredicateConfig` objects, which store raw predicates with no dependencies on other predicates, or
  `DerivedPredicateConfig` objects, which store predicates that build on other predicates.
- `windows`, stored as a dictionary from string window names (which must be unique) to `WindowConfig`
  objects.

Next, we will detail each of these configuration objects.

### Predicates: `DirectPredicateConfig` and `DerivedPredicateConfig`

#### `DirectPredicateConfig`: Configuration of Predicates that can be Computed Directly from Raw Data

These configs consist of the following four fields:

- `code`: The sting value for the categorical code object that is relevant for this predicate. An
  observation will only satisfy this predicate if there is an occurence of this code in the observation.
- `value_min`: If specified, an observation will only satisfy this predicate if the occurrence of the
  underlying `code` with a reported numerical value that is either greater than or greater than or equal to
  `value_min` (with these options being decided on the basis of `value_min_inclusive`, where
  `value_min_incusive=True` indicating that an observation satisfies this predicate if its value is greater
  than or equal to `value_min`, and `value_min_inclusive=False` indicating a greater than but not equal to
  will be used.
- `value_max`: If specified, an observation will only satisfy this predicate if the occurrence of the
  underlying `code` with a reported numerical value that is either less than or less than or equal to
  `value_max` (with these options being decided on the basis of `value_max_inclusive`, where
  `value_max_incusive=True` indicating that an observation satisfies this predicate if its value is less
  than or equal to `value_max`, and `value_max_inclusive=False` indicating a less than but not equal to
  will be used.
- `value_min_inclusive`: See `value_min`
- `value_max_inclusive`: See `value_max`

A given observation will be gauged to satisfy or fail to satisfy this predicate in one of two ways, depending
on its source format.

1. If the source data is in [MEDS](https://github.com/Medical-Event-Data-Standard/meds) format
   (recommended), then the `code` will be checked directly against MEDS' `code` field and the `value_min`
   and `value_max` constraints will be compared against MEDS' `numerical_value` field.
2. If the source data is in [ESGPT](https://eventstreamml.readthedocs.io/en/latest/) format, then the
   `code` will be interpreted in the following manner:
   a. If the code contains a `"//"`, it will be interpreted as being a two element list joined by the
   `"//"` character, with the first element specifying the name of the ESGPT measurement under
   consideration, which should either be of the multi-label classification or multivariate regression
   type, and the second element being the name of the categorical key corresponding to the code in
   question within the underlying measurement specified. If either of `value_min` and `value_max` are
   present, then this measurement must be of a multivariate regression type, and the corresponding
   `values_column` for extracting numerical observations from ESGPT's `dynamic_measurements_df` will be
   sourced from the ESGPT dataset configuration object.
   b. If the code does not contain a `"//"`, it will be interpreted as a direct measurement name that must
   be of the univariate regression type and its value, if needed, will be pulled from the corresponding
   column.

#### `DerivedPredicateConfig`: Configuration of Predicates that Depend on Other Predicates

These confiuration objects consist of only a single string field--`expr`--which contains a limited grammar of
accepted operations that can be applied to other predicates, containing precisely the following:

- `and(pred_1_name, pred_2_name, ...)`: Asserts that all of the specified predicates must be true.
- `or(pred_1_name, pred_2_name, ...)`: Asserts that any of the specified predicates must be true.

Note that, currently, `and`s and `or`s cannot be nested. Upon user request, we may support further advanced
analytic operations over predicates.

### Windows: `WindowConfig`

```
    start: str
    end: str
    start_inclusive: bool
    end_inclusive: bool
    has: dict[str, str]
```
