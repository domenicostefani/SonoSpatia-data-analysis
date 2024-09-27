
import numpy as np

EPSILON = 1e-3

def get_point_list_from_automation(automation_dict):
    assert type(automation_dict) == dict, 'automation_dict must be a dict'
    # print(automation)
    time = np.array([a[0] for a in automation_dict['automation']])
    value = np.array([a[1] for a in automation_dict['automation']])
    shape = np.array([a[2] for a in automation_dict['automation']])

    assert len(time) == len(value) == len(shape), 'Time, value, and shape must have the same length'

    # Ensure that time is monotonically increasing
    assert np.all(np.diff(time) >= 0), 'Time must be monotonically increasing'

    X_toplot = []
    Y_toplot = []

    squarepoint_add = None # When encountering a square shape, this is set to the y value so that, on the next iteration, a point is added at the same x value to create a square shape like Reaper does
    for t,v,s in zip(time,value,shape):
        # First, if we have a square shape, we add the point at the same x value of this next point
        if squarepoint_add is not None:
            if np.isclose(t, X_toplot[-1]):
                variable_epsilon = 0.0
            elif EPSILON > (t - X_toplot[-1]):
                variable_epsilon = (t - X_toplot[-1]) / 2.0
            else:
                variable_epsilon = EPSILON
            X_toplot.append(t - variable_epsilon)
            Y_toplot.append(squarepoint_add)
            squarepoint_add = None

        X_toplot.append(t)
        Y_toplot.append(v)
        if s == 1:
            squarepoint_add = v
        elif s != 0:
            raise ValueError('Shape must be 0 or 1')
        
    X_toplot, Y_toplot = sanitize_time_axis(X_toplot, Y_toplot)

    X_toplot, Y_toplot = add_zero_point(X_toplot, Y_toplot) # Add a zero point at the beginning of the automation (if missing) with value equal to the first point found


    # Ensure that time is still monotonically increasing
    assert np.all(np.diff(X_toplot) >= 0), 'Time must be monotonically increasing even after adding epsilon'

    return X_toplot, Y_toplot


def sanitize_time_axis(X_toplot, Y_toplot):
    # Remove points where X is the same as the next point

    X_res = []
    Y_res = []

    last_x = -1.0
    assert len(X_toplot) == len(Y_toplot), 'X and Y must have the same length'
    for idx,(x,y) in enumerate(zip(X_toplot, Y_toplot)):
        if np.isclose(last_x, x):
            # print('Skipping point with x =', x)
            # Pop last x and y
            X_res.pop()
            Y_res.pop()
        
        X_res.append(x)
        Y_res.append(y)
        last_x = x

    return X_res, Y_res


def add_zero_point(X_toplot, Y_toplot):
    # Add a zero point at the beginning of the automation
    if X_toplot[0] != 0.0:
        X_toplot.insert(0, 0.0)
        Y_toplot.insert(0, Y_toplot[0])
    return X_toplot, Y_toplot