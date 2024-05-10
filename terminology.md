# Algorithm Design and Terminology

## Introduction

We will assume that we are given a dataframe `df` which details events that have happened to subjects. Each
row in the dataframe will have a `subject_id` column which identifies the subject, and a `timestamp` column
which identifies the time at which the event the row is describing happened. We will assume this dataframe has
a collection of columns which describe the event in a variety of ways, limited to counting the number of times
that certain properties hold within each row's event. We will call these additional properties/columns
"predicates" over the events, as they can often be interpreted as boolean or count functions over the event.

For example, we may consider a dataframe where we quantify clinical events happening to medical patients, with
predicates `is_admission`, `is_discharge`, `is_death`, and `is_covid_dx`, looking like this:

| `subject_id` | `timestamp`         | `is_admission` | `is_discharge` | `is_death` | `is_covid_dx` |
| ------------ | ------------------- | -------------- | -------------- | ---------- | ------------- |
| 1            | 2020-01-01 12:03:31 | 1              | 0              | 0          | 0             |
| 1            | 2020-01-01 12:33:01 | 0              | 0              | 0          | 0             |
| 1            | 2020-01-01 13:02:58 | 0              | 0              | 0          | 0             |
| 1            | 2020-01-01 15:00:00 | 0              | 0              | 0          | 0             |
| 1            | 2020-01-04 11:12:00 | 0              | 1              | 0          | 0             |
| 1            | 2022-04-22 07:45:00 | 0              | 0              | 1          | 0             |
| 2            | 2020-01-01 12:03:31 | 1              | 0              | 0          | 0             |
| 2            | 2020-01-02 10:18:29 | 0              | 0              | 0          | 0             |
| 2            | 2020-01-02 16:18:29 | 0              | 0              | 0          | 1             |
| 2            | 2020-01-03 14:47:31 | 0              | 0              | 1          | 0             |
| 3            | 2020-01-01 12:03:31 | 1              | 0              | 0          | 0             |
| 3            | 2020-01-02 12:03:31 | 0              | 1              | 0          | 0             |
| 3            | 2022-01-01 12:03:31 | 1              | 0              | 0          | 0             |
| 3            | 2022-01-06 12:03:31 | 0              | 0              | 1          | 0             |

In this case, we have 3 subjects, which have the respective approximate timeseries of events:

- Subject 1 is admitted, has 3 events that don't satisfy any predicates, is discharged, dies, and has no
  further events.
- Subject 2 is admitted, has an event that satisfies no predicate, has a covid diagnosis, dies, and has no
  further events.
- Subject 3 is admitted, then is discharged, is admitted again, and then dies.

Given data like this, our algorithm is designed to extract valid start and end times of "windows" within a
subject's timeseries that satisfy certain inclusion and exclusion criteria and are defined over temporal and
event-bounded constraints. We can use this algorithm to automatically extract windows of interest from the
record, including but not limited to data cohorts and downstream task labeled datasets for machine learning
applications.

We will specify these windows using a configuration file language that ultimately is interpreted into a tree
structure. For example, suppose we wish to extract a dataset for prediction of in-hospital mortality from the
data defined above, such that we wish to include data the first 24 hours of data of each hospitalization for
input to the model, and predict whether the patient will die in the hospital, subject to the constraints that
the admission in question must be at least 48 hours in length and that the patient must not have a covid
diagnosis within that admission.

We might specify these windows as follows:

```yaml
trigger: is_admission
gap:
  start: trigger
  end: trigger + 48h
  excludes:
    - is_discharge
    - is_death
    - is_covid_dx
input:
  end: trigger + 24h
target:
  start: gap.end
  end: is_discharge | is_death
  label: is_death
  excludes:
    - is_covid_dx
```

Note that this is pseudocode, not the actual configuration language, which may look slightly different. But,
nevertheless, we can see that this set of specifications can be realized in a "valid" form for a patient if
there exist a set of timepoints such that within 48 hours after an admission, there are no discharges, deaths,
or covid diagnosis events, and such that there exists a discharge or death event after the first 48 hours of
an admission such that there are no covid diagnosis events between the end of that first 48 hours and the
subsequent discharge or death event.

Note that these windows form a naturally hierarchical, tree-based structure based on their relative
dependencies on one another. In particular, we can realize the following tree structure for the above
configuration:

```
- Trigger
  - Gap start (trigger)
    - Gap end (Gap start + 48h)
      - Target Start (Gap end)
        - Target End (subsequent `"is_discharge"` or `"is_death"`)
  - Input End
```

Our algorithm will naturally rely on this hierarchical structure by performing a set of recursive database
search operations to extract the windows that satisfy the constraints of the configuration file, recursing
over each subtree to find windows that satisfy the constraints of those subtrees individually.

In the rest of this document, we will detail how our algorithm to automatically extract records that meet
these criteria works, the terminology we use to describe our algorithm (here and in the raw source code and
code comments), the limitations of this algorithm and kinds of tasks it cannot express, and the true
configuration language that we use in practice to specify these windows.

### Terminology

##### Event

An "event" in our dataset is a unique timestamp that occurs for a given subject.

##### Predicate

A "predicate" is a boolean or count function that can be applied to an event to describe the observations that
an underlying dataset included within the timestamp of that event. They will often be boolean functions at the
beginning of the process, but become aggregated into count functions when summarizing windows, so will be
thought of as count functions to capture this generality throughout the algorithm as it rarely if ever is
necessary to distinguish between the two.

##### Window

A "window" is just a time range capturing some portion of a subject's record. It can be inclusive or exclusive
on either endpoint, and may or may not have endpoints corresponding to an extant event in the dataset, as
opposed to a timepoint at which time no event occurred.

##### "Root" of a Subtree

A subtree in the hierarchy of constraint windows has a "root" node in the tree, which corresponds to the start
or end of a window in the set of constraints. For example, the "gap end" node in the tree above is the root of
the subtree `Gap end -> Target Start -> Target End`.

##### A "realized" subtree of constraint windows

A subtree in the hierarchy of constraint windows can be _realized_ in a patient dataset by finding a set of
timestamps such the windows of events they bound satisfy the constraints of the subtree.

##### "Anchor" or "Anchor Event" of a Subtree

A subtree in the hierarchy of constraint windows that can be realized in a real patient's record will have one
most recent ancestor node whose timestamp will correspond to the timestamp of a real event in the patient
record in any realized subtree. This node is called the "anchor" of the subtree. For example, in any
realization of the tree above, the admission event matched by the "trigger" node will be the anchor of the
realization of the `Gap end -> ...` subtree, as the gap end is defined via a relative time gap to the
admission event and thus will not guaranteeably correspond to an extant event in the patient record, whereas
the admission event of the `Trigger` node will always exist in the dataset proper. The notion of an _anchor_
will be useful in the algorithm proper as it will correspond to rows form which we will run database temporal
group-bys and event-based aggregations to extract the windows that satisfy the constraints of the subtree.

## Algorithm Design

### Initialization

### Recursive Step

#### Inputs

In our recursive step, we will be given the following inputs:

##### `predicates_df`

The `predicates_df` dataframe will contain all events and their predicates. This will not be modified across
recursive steps.

##### `subtree_anchor_to_subtree_root_df`

The `subtree_anchor_to_subtree_root_df` dataframe will contain rows corresponding to the timestamps of a
superset of all possible valid anchor events for realizations of the subtree over which we are recursing (a
superset as if there exist no valid realizations of subtrees, then a prospective anchor would be invalid -- if
we can find a valid subtree realization for a prospective anchor in this input dataframe, said anchor would be
a true valid anchor).

This dataframe will also contain the counts of predicates between the prospective anchor events indexed by the
rows of this dataframe and the corresponding possible root timestamps of the subtree over which we are
recursing. This information will be necessary to compute the proper window counts during the recursive step.

##### `offset`

In the event that the subtree root timestamp is not the same as the subtree anchor timestamp (there may be a
temporal offset between the two), the `offset` will be the difference between the two timestamps. If the two
are not the same, they will guaranteeably be separated by a constant offset because the subtree root will
either correspond to a fixed time delta from the subtree anchor or will be a proper event itself, in which
case it will be the subtree anchor.

#### Computation

In the recursive step, we will iterate over all children of the subtree root node, and for each child, we will
do the following:

##### Aggregate Predicates over the Relevant Window

First, we will aggregate the predicates from `predicates_df` over the rows corresponding to the time window
spanning the root of the subtree to the root of the selected child. This aggregation step will always return a
dataframe keyed by the `subject_id` column as well as by any possible prospective realizations of anchor
events for the _subtree rooted at the selected child node_. This computation will take one of two forms:

###### Temporal Aggregation

If the edge linking the subtree root to the child is a temporal relationship (e.g., in our example above, the
"gap.end" node is defined as a fixed time delta from the "gap.start" node), we will aggregate the predicates
by using a "rolling" or "temporal" group-by operation on the `predicates_df` dataframe, summarizing time
windows of the appropriate size and grouping by the `subject_id` column. We will perform this aggregation
globally over the predicates dataframe leveraging the determined edge time delta and the passed offset
parameter (such that we compute the aggregation over the correct window in time from any possible realization
of the subtree anchor) and then filter the resulting dataframe to only include rows corresponding to said
possible subtree anchors. As a temporal edge means the anchor of the child subtree is the same as the anchor
of the passed subtree, this suffices for our intended computation, and we can return it directly.

###### Event-bound Aggregation

If the edge linking the subtree root to the child is an event-bound relationship (e.g., in our example above,
the "target.end" node is defined as the first subsequent `"is_discharge"` or `"is_death"` event after the
"target.start" node), we will aggregate the predicates by using a custom row-predicate-bound aggregation over
the database that will be implemented using differences of cumulative sums within the global `predicates_df`
dataframe. In particular, we will first construct the following three dataframes from our inputs:

1. A dataframe that contains the cumultative count of all predicates seen up until each event (row) in the
   `predicates_df`.
2. A dataframe that contains nulls in each row that does not correspond to a possible prospective
   realization of a child anchor event given the edge constraints and the possible prosepctive subtree
   anchor events and the specified offset, and contains a `True` value otherwise.
3. A dataframe that contains nulls in each row that does not correspond to a possible prospective
   realization of this subtree's anchor event and contains a `True` value otherwise.

From these three dataframes, we can then forward fill the cumulative counts of each predicate seen at each
prospective subtree anchor event forward in time up to the next subsequent possible child anchor node, and
take the difference between the two to compute the relative counts for each predicate column between each
successive pair of subtree anchor events and child anchor events, keyed by child anchor. We must then also
subtract the counts seen between the subtree anchor events and the subtree root events to ensure we are
capturing only the events between the proper subtree root and child root.

##### Filter on Constraints

Next, with these new window counts, we can validate that any inclusion or exclusion criteria are upheld, and
if not remove those realizations as possible realizations of the window before proceeding to the next
computational step.

##### Recurse through the child subtree

With this filtered set of possible prospective child anchor nodes, we can now recurse through the child
subtree.

## Examples

TODO

## Limitations of the Algorithm

TODO

## Configuration Language

TODO
