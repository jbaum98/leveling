import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import NamedTuple, Tuple, List, Any, Dict
from functools import reduce
from operator import add, mul
from itertools import repeat, chain


class Activity(NamedTuple):
    name: str
    capacity: int


class Camper(NamedTuple):
    name: str
    bunk: str
    prefs: Tuple[Activity, Activity, Activity]

    def pref_of(self, activity):
        try:
            index = self.prefs.index(activity)
        except ValueError:
            index = len(self.prefs)
        finally:
            return index + 1


ACTIVITIES: List[Activity] = [
    Activity("Archery", 12),
    Activity("Boating", 4),
    Activity("Improv", 10),
    Activity("Mitbachon", 6),
    Activity("Nagarut", 8),
    Activity("Video", 6),
    Activity("Tzilum", 6),
    Activity("Lyrical", 15),
    Activity("Omanut", 6),
    Activity("Radio", 3),
]

for activity in ACTIVITIES:
    globals()[activity.name.lower()] = activity

CAMPERS: List[Camper] = [
    Camper("Matan", "A15", (archery, boating, nagarut, mitbachon)),
    Camper("Maleia", "A9", (archery, mitbachon, boating, lyrical)),
    Camper("Benjamin", "A13", (archery, mitbachon, tzilum, video)),
    Camper("Edward", "A13", (archery, nagarut, boating, video)),
    Camper("Joshua", "A14", (boating, archery, lyrical, video)),
    Camper("Judah", "A14", (boating, archery, lyrical, mitbachon)),
    Camper("Melilah", "A9", (boating, archery, mitbachon, nagarut)),
    Camper("Zachary", "A14", (boating, archery, nagarut, lyrical)),
    Camper("Baer", "A14", (boating, archery, nagarut, tzilum)),
    Camper("Lucas", "A14", (boating, lyrical, archery, nagarut)),
    Camper("Zeke", "A14", (boating, lyrical, archery, nagarut)),
    Camper("Joe", "A14", (boating, lyrical, nagarut, archery)),
    Camper("Samuel", "A13", (boating, nagarut, archery, video)),
    Camper("Samantha", "A9", (boating, nagarut, lyrical, archery)),
    Camper("Monica", "A7", (boating, nagarut, video, lyrical)),
    Camper("Abigail", "A7", (boating, nagarut, video, omanut)),
    Camper("Annie", "A7", (boating, video, archery, improv)),
    Camper("Shayna", "A7", (boating, video, archery, nagarut)),
    Camper("Noam", "A14", (boating, video, archery, nagarut)),
    Camper("Mikayla", "A7", (improv, mitbachon, archery, lyrical)),
    Camper("Chloe", "A7", (improv, video, archery, nagarut)),
    Camper("Elijah", "A15", (mitbachon, archery, nagarut, video)),
    Camper("Stella", "A9", (mitbachon, archery, nagarut, boating)),
    Camper("Samuel", "A15", (mitbachon, archery, nagarut, lyrical)),
    Camper("Shayne", "A15", (mitbachon, archery, tzilum, video)),
    Camper("Violet", "A7", (mitbachon, boating, lyrical, video)),
    Camper("Jeremy", "A13", (mitbachon, boating, nagarut, tzilum)),
    Camper("Chase", "A13", (mitbachon, boating, nagarut, improv)),
    Camper("Avi", "A15", (mitbachon, boating, nagarut, omanut)),
    Camper("Clara", "A9", (mitbachon, lyrical, boating, nagarut)),
    Camper("Galit", "A7", (mitbachon, lyrical, nagarut, omanut)),
    Camper("Sienna", "A9", (mitbachon, nagarut, tzilum, archery)),
    Camper("Dalia", "A9", (mitbachon, nagarut, tzilum, improv)),
    Camper("Liana", "A9", (mitbachon, omanut, nagarut, boating)),
    Camper("Milli", "A7", (mitbachon, radio, video, omanut)),
    Camper("Jeremy", "A15", (mitbachon, tzilum, nagarut, archery)),
    Camper("Alden", "A15", (mitbachon, video, archery, boating)),
    Camper("Adin", "A13", (mitbachon, video, nagarut, tzilum)),
    Camper("Sarah", "A7", (mitbachon, video, omanut, nagarut)),
    Camper("Ezra", "A13", (nagarut, archery, mitbachon, boating)),
    Camper("Arthur", "A13", (nagarut, boating, tzilum, archery)),
    Camper("Matthew", "A14", (nagarut, boating, video, tzilum)),
    Camper("Galit", "A9", (nagarut, boating, video, omanut)),
    Camper("Batya", "A7", (nagarut, improv, video, archery)),
    Camper("Caleb", "A14", (nagarut, lyrical, boating, archery)),
    Camper("Jamie", "A9", (nagarut, mitbachon, archery, video)),
    Camper("Max", "A15", (nagarut, mitbachon, archery, omanut)),
    Camper("Aiden", "A15", (nagarut, mitbachon, tzilum, video)),
    Camper("Sam", "A15", (nagarut, omanut, archery, mitbachon)),
    Camper("Eitan", "A13", (nagarut, tzilum, boating, video)),
    Camper("Molly", "A9", (omanut, mitbachon, boating, nagarut)),
    Camper("Harrison", "A13", (tzilum, nagarut, boating, nagarut)),
    Camper("Emily", "A9", (video, nagarut, omanut, boating)),
    Camper("Rae", "A9", (video, nagarut, omanut, boating)),
    Camper("Zvi", "A15", (video, nagarut, boating, omanut)),
]


def concat(lists):
    return reduce(chain, lists)


def to_cost_matrix(activities: List[Activity], campers: List[Camper]) -> Any:
    activity_slots = list(concat(repeat(a, a.capacity) for a in activities))

    def cost_of(i, j):
        return campers[i].pref_of(activity_slots[j]) ** 2

    cost_matrix = np.fromfunction(
        np.vectorize(cost_of), (len(campers), len(activity_slots)), dtype=int
    )

    return activity_slots, cost_matrix


def assign(
    activities: List[Activity], campers: List[Camper]
) -> Dict[Camper, Activity]:
    activity_slots, cost_matrix = to_cost_matrix(activities, campers)
    _, assignments = linear_sum_assignment(cost_matrix)
    return {c: activity_slots[i] for c, i in zip(campers, assignments)}
