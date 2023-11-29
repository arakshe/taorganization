import copy
import random as rnd
import numpy as np
import time
import pandas as pd
from functools import reduce

DF_TAS = pd.read_csv("tas.csv")
DF_SECTIONS = pd.read_csv("sections.csv")

class Evo:

    def __init__(self):
        self.pop = {}       # eval -> solution   eval = ((name1, val1), (name2, val2)..)
        self.fitness = {}   # name -> function
        self.agents = {}    # name -> (operator, num_solutions_input)

    def add_fitness_criteria(self, name, f):
        self.fitness[name] = f

    def add_agent(self, name, op, k=1):
        self.agents[name] = (op, k)

    def add_solution(self, sol):
        eval = tuple([(name, f(sol)) for name, f in self.fitness.items()])
        self.pop[eval] = sol

    def get_random_solutions(self, k=1):
        popvals = tuple(self.pop.values())
        return [copy.deepcopy(rnd.choice(popvals)) for _ in range(k)]

    def run_agent(self, name):
        op, k = self.agents[name]
        picks = self.get_random_solutions(k)
        new_solution = op(picks)
        self.add_solution(new_solution)

    @staticmethod
    def _dominates(p, q):
        """ Return whether p dominates q """
        pscores = [score for _, score in p]
        qscores = [score for _, score in q]
        score_diffs = list(map(lambda x, y: y - x, pscores, qscores))
        min_diff = min(score_diffs)
        max_diff = max(score_diffs)
        return min_diff >= 0.0 and max_diff > 0.0

    @staticmethod
    def _reduce_nds(S, p):
        return S - {q for q in S if Evo._dominates(p, q)}

    def remove_dominated(self):
        nds = reduce(Evo._reduce_nds, self.pop.keys(), self.pop.keys())
        self.pop = {k: self.pop[k] for k in nds}

    def evolve(self, n=1, dom=100, time_limit=600 ):

        agent_names = list(self.agents.keys())

        for i in range(n):
            start_time = time.time()
            # pick an agent
            pick = rnd.choice(agent_names)

            # run the agent to produce a new solution
            self.run_agent(pick)

            # periodically cull the population
            # discard dominated solutions
            if i % dom == 0:
                self.remove_dominated()

            elapsed_time = time.time() - start_time
            
            if elapsed_time == time_limit:
                break

        self.remove_dominated()

        return elapsed_time
    
    def __str__(self):
        """ Output the solutions in the population """
        rslt = ""
        for eval, sol in self.pop.items():
            rslt += str(dict(eval)) + ":\t" + str(sol) + "\n"
        return rslt


def overallocation(ta_data):
    """
    Calculate the total overallocation score for TA assignments.
    Args:
        ta_data (numpy.ndarray): An array representing TA assignments to sections.
    Returns:
        int: The total overallocation penalty for all TAs.
    """
    
    max_assigned = DF_TAS["max_assigned"].to_numpy()
    overallocation_penalty = sum(max(0, sum(assignments) - max_ta) for assignments, max_ta in zip(ta_data, max_assigned))
    return overallocation_penalty


def conflicts(ta_data):
    """
    Calculate the number of scheduling conflicts in TA assignments.
    Args:
        ta_data (numpy.ndarray): An array representing TA assignments to sections.
    Returns:
        int: The total number of conflicts across all TAs.
    """
    # Mapping of section times to numerical values
    time_mapping = {'R 1145-125': 1, 'W 950-1130': 2, 'W 1145-125': 3, 'W 250-430': 4, 'W 440-630': 5, 'R 950-1130': 6, 'R 250-430': 7}
    section_times = DF_SECTIONS["daytime"].replace(time_mapping).to_numpy()

    conflict_score = 0
    for assignments in ta_data:
        assigned_times = section_times[assignments == 1]
        unique_times = set(assigned_times)
        if len(unique_times) < len(assigned_times):
            conflict_score += 1

    return conflict_score


def undersupport(ta_data):
    """
    Calculate the total undersupport score for TA assignments.
    Args:
        ta_data (numpy.ndarray): An array representing TA assignments to sections.
    Returns:
        int: The total undersupport penalty across all sections.
    """
    min_ta_required = DF_SECTIONS['min_ta'].tolist()

    undersupport_score = 0
    for section_idx, min_ta in enumerate(min_ta_required):
        assigned_tas = sum(ta_assignment[section_idx] for ta_assignment in ta_data)
        if assigned_tas < min_ta:
            undersupport_score += min_ta - assigned_tas

    return undersupport_score


def unwilling(ta_data):
    """
    Calculate the score for TAs assigned to sections they are unwilling to support.
    Args:
        ta_data (numpy.ndarray): An array representing TA assignments to sections.
    Returns:
        int: The total count of unwilling assignments.
    """
    preference_values = {'U': 2, 'W': 1, 'P': 0}
    preferences = [[preference_values[pref] for pref in row[3:]] for row in DF_TAS.itertuples(index=False)]

    unwilling_count = sum(
        1 for ta_idx, ta_prefs in enumerate(preferences)
        for section_idx, assigned in enumerate(ta_data[ta_idx])
        if assigned == 1 and ta_prefs[section_idx] == 2
    )

    return unwilling_count


def unpreferred(ta_data):
    """
    Calculate the count of TAs assigned to sections that are not their preferred choices.
    Args:
        ta_data (numpy.ndarray): An array representing TA assignments to sections.
    Returns:
        int: The total count of unpreferred assignments.
    """
    preference_values = {'U': 2, 'W': 1, 'P': 0}
    preferences = [[preference_values[pref] for pref in row[3:]] for row in DF_TAS.itertuples(index=False)]

    unpreferred_count = sum(
        1 for ta_idx, ta_prefs in enumerate(preferences)
        for section_idx, assigned in enumerate(ta_data[ta_idx])
        if assigned == 1 and ta_prefs[section_idx] == 1
    )

    return unpreferred_count

# Define agents

def random_reassignment_agent(current_solution):
    """
    An agent that randomly reassigns a TA to a new section.
    Args:
        current_solution (list of lists): The current assignment solution.
    Returns:
        list of lists: A new solution with one TA reassigned.
    """
    new_solution = copy.deepcopy(current_solution)
    num_tas = len(new_solution)
    num_sections = len(new_solution[0])

    # Randomly select a TA
    ta_idx = rnd.randrange(num_tas)

    # Randomly select a new section for this TA
    new_section = rnd.randrange(num_sections)

    # Assign TA to the new section
    new_solution[ta_idx] = [0] * num_sections  # Remove TA from all sections
    new_solution[ta_idx][new_section] = 1  # Assign TA to the new section

    return new_solution

def swap_sections_agent(current_solution):
    """
    An agent that swaps sections between two randomly chosen TAs.
    Args:
        current_solution (list of lists): The current assignment solution.
    Returns:
        list of lists: A new solution with swapped sections.
    """
    new_solution = copy.deepcopy(current_solution)
    num_tas = len(new_solution)
    num_sections = len(new_solution[0])

    # Randomly select two different TAs
    ta_idx1, ta_idx2 = rnd.sample(range(num_tas), 2)

    # Randomly select a section for each TA
    section1 = rnd.randrange(num_sections)
    section2 = rnd.randrange(num_sections)

    # Swap the sections between the two TAs
    new_solution[ta_idx1][section1], new_solution[ta_idx2][section2] = new_solution[ta_idx2][section2], new_solution[ta_idx1][section1]

    return new_solution

def find_best_sol(E):
    """
    Finds the solution in the population with the lowest total objective score.
    
    Args:
        E (Evo object): Instance of the Evo class containing the population.
    Returns:
        best_solution (dict): A dictionary containing the best solution, its score, and the objectives.
    """
    best_score = float('inf')  # more readable initialization of the best score.
    best_solution = None

    # Iterate through all solutions in the population.
    for evaluation, solution in E.pop.items():
        # Calculate the total score for the current solution.
        total_score = sum(score for _, score in evaluation)

        # Update the best solution if the current one has a lower score.
        if total_score < best_score:
            best_score = total_score
            best_solution = {
                'score': best_score,
                'solution': solution,
                'objectives': dict(evaluation)
            }

    return best_solution

# Test
def cases():
    test1 = np.array(pd.read_csv('test1.csv', header=None))
    test2 = np.array(pd.read_csv('test2.csv', header=None))
    test3 = np.array(pd.read_csv('test3.csv', header=None))
    return test1, test2, test3

def test_overallocation(case1, case2, case3):
    assert overallocation(case1) == 37, "Incorrect overallocation score for test1"
    assert overallocation(case2) == 41, "Incorrect overallocation score for test2"
    assert overallocation(case3) == 23, "Incorrect overallocation score for test3"
    print('overallocation tests passed')

def test_conflicts(case1, case2, case3):
    assert conflicts(case1) == 8, "Incorrect conflicts score for test1"
    assert conflicts(case2) == 5, "Incorrect conflicts score for test2"
    assert conflicts(case3) == 2, "Incorrect conflicts score for test3"
    print('conflicts tests passed')
def test_undersupport(case1, case2, case3):
    assert undersupport(case1) == 1, "Incorrect undersupport score for test1"
    assert undersupport(case2) == 0, "Incorrect undersupport score for test2"
    assert undersupport(case3) == 7, "Incorrect undersupport score for test3"
    print('undersupport tests passed')
def test_underwilling(case1, case2, case3):
    assert unwilling(case1) == 53, "Incorrect unwilling score for test1"
    assert unwilling(case2) == 58, "Incorrect unwilling score for test2"
    assert unwilling(case3) == 43, "Incorrect unwilling score for test3"
    print('unwilling tests passed')

def test_unpreferred(case1, case2, case3):
    assert unpreferred(case1) == 15, "Incorrect unpreferred score for test1"
    assert unpreferred(case2) == 19, "Incorrect unpreferred score for test2"
    assert unpreferred(case3) == 10, "Incorrect unpreferred score for test3"
    print('unpreferred tests passed')

def cases():
    test1 = np.array(pd.read_csv('test1.csv', header=None))
    test2 = np.array(pd.read_csv('test2.csv', header=None))
    test3 = np.array(pd.read_csv('test3.csv', header=None))
    return test1, test2, test3

def main():
    E = Evo()

    # Register objectives
    E.add_fitness_criteria("overallocation", overallocation)
    E.add_fitness_criteria("conflicts", conflicts)
    E.add_fitness_criteria("undersupport", undersupport)
    E.add_fitness_criteria("unwilling", unwilling)
    E.add_fitness_criteria("unpreferred", unpreferred)

    # Register agents
    E.add_agent("swapper", swapper, 1)

    # Generate and add random initial solutions to the environment
    for _ in range(10):
        # Generate a random solution
        rand_solution = (np.random.rand(len(DF_TAS), len(DF_SECTIONS)) > 0.9).astype(int)
        # Add solution to the environment
        E.add_solution(rand_solution)

    # Run the optimizer for 10 minutes (600 seconds)
    E.evolve(time_limit=600)

    # Report best solutions
    find_best_sol(E)
    
    best_solution = find_best_sol(E)

    # Extracting and displaying the best score, objectives, and array
    best_score = best_solution['score']
    best_objectives = best_solution['objectives']
    best_array = best_solution['solution']

    print("Best Score:", best_score)
    print("Best Objectives:", best_objectives)
    print("Best Array:", best_array)

    # Testing the objective functions with provided cases
    case1, case2, case3 = cases()
    test_overallocation(case1, case2, case3)
    test_conflicts(case1, case2, case3)
    test_undersupport(case1, case2, case3)
    test_unpreferred(case1, case2, case3)
    test_underwilling(case1, case2, case3)

if __name__ == "__main__":
    main()
