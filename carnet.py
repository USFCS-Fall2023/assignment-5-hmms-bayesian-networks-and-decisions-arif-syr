from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

car_model = BayesianNetwork(
    [
        ("Battery", "Radio"),
        ("Battery", "Ignition"),
        ("Ignition", "Starts"),
        ("Gas", "Starts"),
        ("Starts", "Moves")
    ]
)

# Defining the parameters using CPT
from pgmpy.factors.discrete import TabularCPD

cpd_battery = TabularCPD(
    variable="Battery", variable_card=2, values=[[0.70], [0.30]],
    state_names={"Battery": ['Works', "Doesn't work"]},
)

cpd_gas = TabularCPD(
    variable="Gas", variable_card=2, values=[[0.40], [0.60]],
    state_names={"Gas": ['Full', "Empty"]},
)

cpd_radio = TabularCPD(
    variable="Radio", variable_card=2,
    values=[[0.75, 0.01], [0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Radio": ["turns on", "Doesn't turn on"],
                 "Battery": ['Works', "Doesn't work"]}
)

cpd_ignition = TabularCPD(
    variable="Ignition", variable_card=2,
    values=[[0.75, 0.01], [0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Ignition": ["Works", "Doesn't work"],
                 "Battery": ['Works', "Doesn't work"]}
)

cpd_starts = TabularCPD(
    variable="Starts",
    variable_card=2,
    values=[[0.95, 0.05, 0.05, 0.001], [0.05, 0.95, 0.95, 0.9999]],
    evidence=["Ignition", "Gas"],
    evidence_card=[2, 2],
    state_names={"Starts": ['yes', 'no'], "Ignition": ["Works", "Doesn't work"], "Gas": ['Full', "Empty"]},
)

cpd_moves = TabularCPD(
    variable="Moves", variable_card=2,
    values=[[0.8, 0.01], [0.2, 0.99]],
    evidence=["Starts"],
    evidence_card=[2],
    state_names={"Moves": ["yes", "no"],
                 "Starts": ['yes', 'no']}
)

# Associating the parameters with the model structure
car_model.add_cpds(cpd_starts, cpd_ignition, cpd_gas, cpd_radio, cpd_battery, cpd_moves)

car_infer = VariableElimination(car_model)

print(car_infer.query(variables=["Moves"], evidence={"Radio": "turns on", "Starts": "yes"}))


def run():
    # Probability that the battery is dead given the car does not move
    battery_given_no_move = car_infer.query(variables=["Battery"], evidence={"Moves": "no"})
    print(battery_given_no_move)
    print(f"Probability that the battery is dead given the car does not move: {battery_given_no_move.values[1]}")

    # Probability the car does not start given the radio does not work
    no_start_given_radio_not_working = car_infer.query(variables=["Starts"], evidence={"Radio": "Doesn't turn on"})
    print(no_start_given_radio_not_working)
    print(f"Probability the car does not start given the radio does not work: {no_start_given_radio_not_working.values[1]}")

    # Change in probability of radio working if gas is full given battery works
    radio_given_battery_works = car_infer.query(variables=["Radio"], evidence={"Battery": "Works"})
    radio_given_battery_works_and_gas = car_infer.query(variables=["Radio"],
                                                        evidence={"Battery": "Works", "Gas": "Full"})
    print(radio_given_battery_works)
    print(radio_given_battery_works_and_gas)
    print(f"Probability of the radio working given the battery works: {radio_given_battery_works.values[0]}")
    print(
        f"Probability of the radio working given the battery works and gas is full: {radio_given_battery_works_and_gas.values[0]}")
    print("No change in probability if gas is full")

    # Change in probability of the ignition failing if we observe that the car does not have gas in it
    # given that the car doesn't move
    ignition_given_no_move = car_infer.query(variables=["Ignition"], evidence={"Moves": "no"})
    ignition_given_no_move_no_gas = car_infer.query(variables=["Ignition"], evidence={"Moves": "no", "Gas": "Empty"})
    print(ignition_given_no_move)
    print(ignition_given_no_move_no_gas)
    print(f"Probability of ignition failing given car does not move: {ignition_given_no_move.values[1]}")
    print(f"Probability of ignition failing given car does not move and gas is empty: {ignition_given_no_move_no_gas.values[1]}")

    # Probability of car starting if radio works and gas is full
    car_starts_given_radio_works_and_gas = car_infer.query(variables=["Starts"],
                                                           evidence={"Radio": "turns on", "Gas": "Full"})
    print(car_starts_given_radio_works_and_gas)


# Adding a new node, and updating cpd_starts
print("Adding new node KeyPresent and updating the starts node in carnet.py")
cpd_key_present = TabularCPD(
    variable="KeyPresent",
    variable_card=2,
    values=[[0.7], [0.3]],
    state_names={"KeyPresent": ['yes', 'no']}
)

cpd_starts = TabularCPD(
    variable="Starts",
    variable_card=2,
    values=[
        [0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],  # Starts=yes
        [0.01, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]   # Starts=no
    ],
    evidence=["Ignition", "Gas", "KeyPresent"],
    evidence_card=[2, 2, 2],
    state_names={
        "Starts": ['yes', 'no'],
        "Ignition": ['Works', "Doesn't work"],
        "Gas": ['Full', "Empty"],
        "KeyPresent": ['yes', 'no']
    }
)
