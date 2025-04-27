## Configuration Language Specification

This document specifies the configuration language for the automatic extraction of task dataframes and cohorts
from structured EHR data organized either via the [MEDS](https://github.com/Medical-Event-Data-Standard/meds)
format (recommended) or the [ESGPT](https://eventstreamml.readthedocs.io/en/latest/) format. This extraction
system works by defining a configuration object that details the underlying concepts, inclusion/exclusion, and
labeling criteria for the cohort/task to be extracted, then using a recursive algorithm to identify all
realizations of valid patient time-ranges of data that satisfy those constraints from the raw data. For more
details on the recursive algorithm, see [Algorithm Design](https://eventstreamaces.readthedocs.io/en/latest/technical.html#algorithm-design).

As indicated above, these cohorts are specified through a combination of concepts (realized as event
_predicate_ functions, _aka_ "predicates") which are _dataset specific_ and inclusion/exclusion/labeling
criteria which, conditioned on a set of predicate definitions, are _dataset agnostic_.

Predicates are currently limited to "count" predicates, which are predicates that count the number of times a
boolean condition is satisfied over a given time window, which can either be a single timestamp, thus tracking
whether how many observations there were that satisfied the boolean condition in that event (_aka_ at that
timestamp) or over 1-dimensional windows. In the future, predicates may expand to include other notions of
functional characterization, such as tracking the average/min/max value a concept takes on over a time-period,
etc.

Constraints are specified in terms of time-points that can be bounded by events that satisfy predicates or
temporal relationships on said events. The windows between these time-points can then either be constrained to
contain events that satisfy certain aggregation functions over predicates for these time frames.

______________________________________________________________________

In the machine form used by ACES, the configuration file consists of three parts:

- `predicates`, stored as a dictionary from string predicate names (which must be unique) to either
  {py:class}`aces.config.PlainPredicateConfig` objects, which store raw predicates with no dependencies on other predicates, or
  {py:class}`aces.config.DerivedPredicateConfig` objects, which store predicates that build on other predicates.
- `trigger`, stored as a string to `EventConfig`
- `windows`, stored as a dictionary from string window names (which must be unique) to {py:class}`aces.config.WindowConfig`
  objects.

Below, we will detail each of these configuration objects.

______________________________________________________________________

### Predicates: `PlainPredicateConfig` and `DerivedPredicateConfig`

#### {py:class}`aces.config.PlainPredicateConfig`: Configuration of Predicates that can be Computed Directly from Raw Data

These configs consist of the following four fields:

- `code`: The string expression for the code object that is relevant for this predicate. An
  observation will only satisfy this predicate if there is an occurrence of this code in the observation.
  The field can additionally be a dictionary with either a `regex` key and the value being a regular
  expression (satisfied if the regular expression evaluates to True), or a `any` key and the value being a
  list of strings (satisfied if there is an occurrence for any code in the list).

  > [!NOTE]
  > Each individual definition of `PlainPredicateConfig` and `code` will generate a separate predicate
  > column. Thus, for memory optimization, it is strongly recommended to match multiple values using either
  > the List of Values or Regular Expression formats whenever possible.

- `value_min`: If specified, an observation will only satisfy this predicate if the occurrence of the
  underlying `code` with a reported numerical value that is either greater than or greater than or equal to
  `value_min` (with these options being decided on the basis of `value_min_inclusive`, where
  `value_min_inclusive=True` indicating that an observation satisfies this predicate if its value is greater
  than or equal to `value_min`, and `value_min_inclusive=False` indicating a greater than but not equal to
  will be used).

- `value_max`: If specified, an observation will only satisfy this predicate if the occurrence of the
  underlying `code` with a reported numerical value that is either less than or less than or equal to
  `value_max` (with these options being decided on the basis of `value_max_inclusive`, where
  `value_max_inclusive=True` indicating that an observation satisfies this predicate if its value is less
  than or equal to `value_max`, and `value_max_inclusive=False` indicating a less than but not equal to
  will be used).

- `value_min_inclusive`: See `value_min`

- `value_max_inclusive`: See `value_max`

- `other_cols`: This optional field accepts a 1-to-1 dictionary of column names to column values, and can be
  used to specify further constraints on other columns (ie., not `code`) for this predicate.

A given observation will be gauged to satisfy or fail to satisfy this predicate in one of two ways, depending
on its source format.

1. If the source data is in [MEDS](https://github.com/Medical-Event-Data-Standard/meds) format
   (recommended), then the `code` will be checked directly against MEDS' `code` field and the `value_min`
   and `value_max` constraints will be compared against MEDS' `numeric_value` field.

   [!NOTE]
   This syntax does not currently support defining predicates that also rely on matching other,
   optional fields in the MEDS syntax; if this is a desired feature for you, please let us know by filing a
   GitHub issue or pull request or upvoting any existing issue/PR that requests/implements this feature,
   and we will add support for this capability.
