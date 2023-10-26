"""
Authors:Damian Kreft, Sebastian Kreft
Required environment: Python3, matplotlib, scikit-fuzzy

Should you use the car - fuzzy logic implementation
App helps you to decide if you should take a car or not, considering weather conditions 
(rain, wind power and temperature - in scale from 0 to 10, fuzzy values for them are good, average, poor).
The output value tells how much you should choose a car (fuzzy values are high, medium, low)

 Rules
    - IF rain chance is high ('good'), wind power is high ('good') and temperature is not optimal ('poor')
     THEN should you ride a car value is high ('high').

    - IF rain chance is average ('average'), wind power is average ('average') and temperature is average ('average')
     THEN should you ride a car value is medium ('medium').

    - IF rain chance is low ('poor'), wind power is low ('poor') and temperature is optimal ('good')
     THEN should you ride a car value is low ('low').


"""
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# New Antecedent/Consequent objects hold universe variables and membership
# functions
rainChance = ctrl.Antecedent(np.arange(0, 11, 1), 'rainChance')
temperature = ctrl.Antecedent(np.arange(0, 11, 1), 'temperature')
windPower = ctrl.Antecedent(np.arange(0, 11, 1), 'windPower')
should_I_drive_a_car = ctrl.Consequent(np.arange(0, 11, 1), 'should_I_drive_a_car', 'centroid')


rainChance.automf(3)
temperature.automf(3)
windPower.automf(3)

#Membership functions

should_I_drive_a_car['low'] = fuzz.trimf(should_I_drive_a_car.universe, [0, 0, 30])
should_I_drive_a_car['medium'] = fuzz.trimf(should_I_drive_a_car.universe, [0, 30, 100])
should_I_drive_a_car['high'] = fuzz.trimf(should_I_drive_a_car.universe, [30, 100, 100])


# poor | average | good 
rule1 = ctrl.Rule(rainChance['good'] | temperature['poor'] | windPower['good'], should_I_drive_a_car['high'])
rule2 = ctrl.Rule(rainChance['average'] | temperature['average'] | windPower['average'], should_I_drive_a_car['medium'])
rule3 = ctrl.Rule(rainChance['poor'] | temperature['good'] | windPower['poor'], should_I_drive_a_car['low'])



car_picking_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])



car_picking = ctrl.ControlSystemSimulation(car_picking_ctrl)

# We specify our inputs here
car_picking.input['rainChance'] = 100
car_picking.input['windPower'] = 100
car_picking.input['temperature'] = 100

car_picking.compute()

# We can view the result as well as visualize it
print(car_picking.output['should_I_drive_a_car'])
should_I_drive_a_car.view(sim=car_picking)
input()
