from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


#task1: simple projectile motion
def compute_trajectory_1(u, thetaDeg, g, h):
    theta = np.deg2rad(thetaDeg)
    dt = 0.01

    vx = u * np.cos(theta)
    vy = u * np.sin(theta)

    xCords = [0]
    yCords= [h]

    x = 0
    y = h
    t = 0
    while y >= 0:
        x += vx * dt
        y += vy * dt - 0.5 * g * dt**2
        vy -= g * dt
        t += dt
        xCords.append(x)
        yCords.append(y)
    
    return xCords, yCords

@app.route('/page1')
def index_1():
    return render_template('task1.html')

@app.route('/update_1', methods=['POST'])
def update_1():
    g = float(request.json['gravity'])
    theta_deg = float(request.json['angle'])
    u = float(request.json['speed'])
    h = float(request.json['height'])
    
    xCords, yCords = compute_trajectory_1(u, theta_deg, g, h)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xCords, y=yCords, mode='lines', name='Trajectory'))
    fig.update_layout(title='Simple Projectile Motion Model',
                      xaxis_title='x (m)',
                      yaxis_title='y (m)',
                      paper_bgcolor='#26252C', 
                      plot_bgcolor='#26252C',
                      font=dict(color='white'))

    graphJSON = pio.to_json(fig)
    return jsonify(graphJSON)

#task2: analytical model
def compute_trajectory_2(g, thetaDeg, u, h):
    theta = np.deg2rad(thetaDeg)
    
    T = (u * np.sin(theta) + np.sqrt((u * np.sin(theta))**2 + 2 * g * h)) / g
    R = u * np.cos(theta) * T
    xa = (u**2 * np.sin(2 * theta)) / (2 * g)
    ya = h + (u**2 * np.sin(theta)**2) / (2 * g)
    
    xCords = np.linspace(0, R, num=500)
    yCords = h + xCords * np.tan(theta) - (g * (1 + np.tan(theta)**2) * xCords**2) / (2 * u**2)
    
    valid_indices = yCords >= 0
    xCords = xCords[valid_indices]
    yCords = yCords[valid_indices]
    
    return xCords, yCords, xa, ya, T

@app.route('/page2')
def index_2():
    return render_template('task2.html')

@app.route('/update_2', methods=['POST'])
def update_2():
    g = float(request.json['gravity'])
    thetaDeg = float(request.json['angle'])
    u = float(request.json['speed'])
    h = float(request.json['height'])
    
    x_values, y_values, xa, ya, T = compute_trajectory_2(g, thetaDeg, u, h)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name='Trajectory'))
    fig.add_trace(go.Scatter(x=[xa], y=[ya], mode='markers', name='Apogee', marker=dict(color='red', size=10)))
    fig.update_layout(title='Exact Projectile Motion Model',
                      xaxis_title='x (m)',
                      yaxis_title='y (m)',
                      paper_bgcolor='#26252C', 
                      plot_bgcolor='#26252C',
                      font=dict(color='white'))
    graphJSON = pio.to_json(fig)
    return jsonify(graphJSON=graphJSON, time_of_flight=T)

#task3

def calculateMinimumLaunchAngleAndSpeed_3(X=1000, Y=300, h=0, g=9.81):
    a = Y - h
    b = np.sqrt(X**2 + (Y - h)**2)
    minUSquared = g * (a + b)
    
    if minUSquared <= 0:
        print("no min speed")
    
    minU = np.sqrt(minUSquared)
    
    theta = np.arctan((a + b) / X)
    thethaDeg = np.degrees(theta)
    return minU, thethaDeg

def calculate_launch_angles_3(X=1000, Y=300, h=0, g=9.81, u=150):

    a = g * X**2 / (2 * u**2)
    b = -X
    c = Y - h + g * X**2 / (2 * u**2)
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        return None, None
    
    discriminant_sqrt = np.sqrt(discriminant)
    angle1Rad = np.arctan((-b + discriminant_sqrt) / (2 * a))
    angle2Rad = np.arctan((-b - discriminant_sqrt) / (2 * a))
    angle1Deg = np.degrees(angle1Rad)
    angle2Deg = np.degrees(angle2Rad)
    lowballAngle = min(angle1Deg, angle2Deg)
    highballAngle = max(angle1Deg, angle2Deg)

    return lowballAngle, highballAngle

def get_trajectory_data_3(angle, u, h, g):

    angleRad = np.radians(angle)
    
    a = 0.5 * g
    b = -u * np.sin(angleRad)
    c = -h

    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return np.array([]), np.array([])  # no trajectory when disc is neg (no real sol)
    
    T = (-b + np.sqrt(discriminant)) / (2 * a)
    
    tMax = np.linspace(0, T, num=100)
    xCords = u * np.cos(angleRad) * tMax
    yCords = h + u * np.sin(angleRad) * tMax - 0.5 * g * tMax**2
    
    mask = yCords >= 0 #for formatting (so graph doesnt go below y = 0)
    return xCords[mask], yCords[mask]

def plot_projectile_trajectories_3(X=1000, Y=300, h=0, g=9.81, u=150):
    lowBallAngle, highBallAngle = calculate_launch_angles_3(X, Y, h, g, u)

    fig = go.Figure()

    if lowBallAngle is not None and highBallAngle is not None:

        xHigh, yHigh = get_trajectory_data_3(highBallAngle, u, h, g)
        fig.add_trace(go.Scatter(x=xHigh, y=yHigh, mode='lines', name=f'High Ball Trajectory ({highBallAngle:.1f}°)'))

        xLow, Llow = get_trajectory_data_3(lowBallAngle, u, h, g)
        fig.add_trace(go.Scatter(x=xLow, y=Llow, mode='lines', name=f'Low Ball Trajectory ({lowBallAngle:.1f}°)'))

    minU, thethaDeg = calculateMinimumLaunchAngleAndSpeed_3(X, Y, h, g)
    xMin, yMin = get_trajectory_data_3(thethaDeg, minU, h, g)
    fig.add_trace(go.Scatter(x=xMin, y=yMin, mode='lines', name=f'Minimum Launch Trajectory ({thethaDeg:.1f}°)'))

    #bouding trajectory
    xBoundCords = np.linspace(0, 10 * X, num=100)
    yBoundCords = h + (u**2) / (2 * g) - (g / (2 * u**2)) * xBoundCords**2
    maskBound = yBoundCords >= 0  # filtering formating
    fig.add_trace(go.Scatter(x=xBoundCords[maskBound], y=yBoundCords[maskBound], mode='lines', name='Bounding Parabola'))

    
    maxHorizontalAngle = np.arcsin(1/np.sqrt(2+2*g*h/(u**2)))
    maxHorizontalAngleDeg = np.degrees(maxHorizontalAngle)
    x_max, y_max = get_trajectory_data_3(maxHorizontalAngleDeg, u, h, g)
    fig.add_trace(go.Scatter(x=x_max, y=y_max, mode='lines', name=f'Maximum Horizontal Path ({maxHorizontalAngleDeg:.1f}°)'))

    fig.add_trace(go.Scatter(x=[X], y=[Y], mode='markers', marker=dict(color='white', size=10), name=f'Target Point ({X}, {Y})'))

    fig.update_layout(
        xaxis_title='Horizontal Distance (m)',
        yaxis_title='Vertical Distance (m)',
                      paper_bgcolor='#26252C', 
                      plot_bgcolor='#26252C',
                      font=dict(color='white'),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        hovermode='closest',
        showlegend=True
    )

    graphJSON = fig.to_json()
    return graphJSON

@app.route('/page3', methods=['GET'])
def index_3():
    plot = plot_projectile_trajectories_3()
    return render_template('task3.html', plot=plot, X=1000, Y=300, h=0, g=9.81, u=150)

@app.route('/update_3', methods=['POST'])
def update_3():
    data = request.json
    X = float(data['X'])
    Y = float(data['Y'])
    h = float(data['h'])
    g = float(data['g'])
    u = float(data['u'])
    plot = plot_projectile_trajectories_3(X, Y, h, g, u)
    return jsonify({'plot': plot})

# Task 4: Arc Length and Time of Flight
def calculateArcLengthOfProjectile(u, theta, g, h):
    thetaRad = np.radians(theta)
    X = (u**2 / g) * (np.sin(thetaRad) * np.cos(thetaRad) + np.cos(thetaRad) * np.sqrt(np.sin(thetaRad)**2 + 2 * g * h / u**2))
    tanTheta = np.tan(thetaRad)
    lowerLimit = tanTheta - (g * X / u**2) * (1 + tanTheta**2)
    upperLimit = tanTheta

    def integralPart(z):
        return 0.5 * np.log(np.abs(np.sqrt(1 + z**2) + z)) + 0.5 * z * np.sqrt(1 + z**2)

    arcLength = ((u**2) / (g * (1 + tanTheta**2))) * (integralPart(upperLimit) - integralPart(lowerLimit))
    return arcLength

def compute_trajectory_4(g, thetaDeg, u, h):
    theta = np.deg2rad(thetaDeg)
    
    T = (u * np.sin(theta) + np.sqrt((u * np.sin(theta))**2 + 2 * g * h)) / g
    R = u * np.cos(theta) * T
    xa = (u**2 * np.sin(2 * theta)) / (2 * g)
    ya = h + (u**2 * np.sin(theta)**2) / (2 * g)
    
    xCords = np.linspace(0, R, num=500)
    yCords = h + xCords * np.tan(theta) - (g * (1 + np.tan(theta)**2) * xCords**2) / (2 * u**2)
    
    filteredCords = yCords >= 0 #filitering
    xCords = xCords[filteredCords]
    yCords = yCords[filteredCords]

    thethaMax = np.arcsin(1 / np.sqrt(2 + 2 * g * h / u**2))
    rMax = (u**2 / g) * (np.sin(2 * thethaMax) + np.cos(thethaMax) * np.sqrt(np.sin(thethaMax)**2 + 2 * g * h / u**2))
    xMaxCords = np.linspace(0, rMax, num=500)
    yMaxCords = h + xMaxCords * np.tan(thethaMax) - (g * (1 + np.tan(thethaMax)**2) * xMaxCords**2) / (2 * u**2)

    filteredMaxCords = yMaxCords >= 0
    xMaxCords = xMaxCords[filteredMaxCords]
    yMaxCords = yMaxCords[filteredMaxCords]
    
    TMax = (u * np.sin(thethaMax) + np.sqrt((u * np.sin(thethaMax))**2 + 2 * g * h)) / g
    
    arcLength = calculateArcLengthOfProjectile(u, thetaDeg, g, h)
    arcLengthMax = calculateArcLengthOfProjectile(u, np.degrees(thethaMax), g, h)
    
    return (xCords, yCords, xa, ya, T, TMax, arcLength, arcLengthMax, xMaxCords, yMaxCords)

@app.route('/page4')
def index_4():
    return render_template('task4.html')

@app.route('/update_4', methods=['POST'])
def update_4():
    g = float(request.json['gravity'])
    theta_deg = float(request.json['angle'])
    u = float(request.json['speed'])
    h = float(request.json['height'])
    
    x_values, y_values, xa, ya, T, T_max, arc_length, arc_length_max, x_values_max, y_values_max = compute_trajectory_4(g, theta_deg, u, h)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name='User-defined Trajectory'))
    fig.add_trace(go.Scatter(x=x_values_max, y=y_values_max, mode='lines', name='Maximum Range Trajectory', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=[xa], y=[ya], mode='markers', name='Apogee', marker=dict(color='red', size=10)))
    fig.update_layout(title='Exact Projectile Motion Model',
                      xaxis_title='x (m)',
                      yaxis_title='y (m)',
                      paper_bgcolor='#26252C', 
                      plot_bgcolor='#26252C',
                      font=dict(color='white'))
    graphJSON = pio.to_json(fig)
    return jsonify(
        graphJSON=graphJSON,
        time_of_flight=round(T, 3),
        max_time_of_flight=round(T_max, 3),
        arc_length=round(arc_length, 3),
        max_arc_length=round(arc_length_max, 3)
    )

#Task7

u = 10  # launch speed in m/s
g = 9.81  # acceleration due to gravity in m/s^2


def calculate_range_7(t, theta):
    return np.sqrt((u**2) * (t**2) - g * (t**3) * u * np.sin(theta) + 0.25 * (g**2) * (t**4))

def calculate_displacements_7(t, theta):
    x = u * t * np.cos(theta)
    y = u * t * np.sin(theta) - 0.5 * g * t**2
    mask = y >= 0  # filter where y is greater than or equal to 0
    return x[mask], y[mask]

# Generate time values
t = np.linspace(0, 5, 500)

angles = [60, 70, 85, 45]
theta_blue = np.deg2rad(angles[0])
theta_green = np.deg2rad(angles[1])
theta_red = np.deg2rad(angles[2])
theta_yellow = np.deg2rad(angles[3])

r_blue = calculate_range_7(t, theta_blue)
r_green = calculate_range_7(t, theta_green)
r_red = calculate_range_7(t, theta_red)
r_yellow = calculate_range_7(t, theta_yellow)

x_blue, y_blue = calculate_displacements_7(t, theta_blue)
x_green, y_green = calculate_displacements_7(t, theta_green)
x_red, y_red = calculate_displacements_7(t, theta_red)
x_yellow, y_yellow = calculate_displacements_7(t, theta_yellow)

# using formula to calculate min max heights times
def compute_min_max_times_7(u, g, angles):
    minMaxData = []
    for angle in angles:
        theta = np.deg2rad(angle)
        if np.sin(theta)**2 >= 8/(9 * 9):
            term = np.sqrt(np.sin(theta)**2 - 8/9)
            t1 = (3*u / (2*g)) * (np.sin(theta) + term)
            t2 = (3*u / (2*g)) * (np.sin(theta) - term)
            minMaxData.append((angle, t1, t2))
    return minMaxData

minMaxTimes = compute_min_max_times_7(u, g, angles)

# plot range vs time
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=t, y=r_blue, mode='lines', name='Angle = 60°', line=dict(color='blue')))
fig1.add_trace(go.Scatter(x=t, y=r_green, mode='lines', name='Angle = 70°', line=dict(color='green')))
fig1.add_trace(go.Scatter(x=t, y=r_red, mode='lines', name='Angle = 85°', line=dict(color='red')))
fig1.add_trace(go.Scatter(x=t, y=r_yellow, mode='lines', name='Angle = 45°', line=dict(color='yellow')))

fig1.update_layout(
    title='', #add title here if needed
    xaxis_title='Time (s)',
    yaxis_title='Range (m)',
    margin=dict(l=50, r=50, t=50, b=50),
)

# plot y vs x
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=x_blue, y=y_blue, mode='lines', name='Angle = 60°', line=dict(color='blue')))
fig2.add_trace(go.Scatter(x=x_green, y=y_green, mode='lines', name='Angle = 70°', line=dict(color='green')))
fig2.add_trace(go.Scatter(x=x_red, y=y_red, mode='lines', name='Angle = 85°', line=dict(color='red')))
fig2.add_trace(go.Scatter(x=x_yellow, y=y_yellow, mode='lines', name='Angle = 45°', line=dict(color='yellow')))

fig2.update_layout(
    title='', #add title here if needed
    xaxis_title='Horizontal Displacement (m)',
    yaxis_title='Vertical Displacement (m)',
    margin=dict(l=50, r=50, t=50, b=50),
)

# Print min max points for X-Y graph (For debugging only)
for angle, t1, t2 in minMaxTimes:
    theta = np.deg2rad(angle)
    x1, y1 = calculate_displacements_7(t1, theta)
    x2, y2 = calculate_displacements_7(t2, theta)
    
    if len(x1) > 0 and len(y1) > 0 and len(x2) > 0 and len(y2) > 0:
        print(f"Angle = {angle}°, Point 1: ({x1[0]}, {y1[0]}), Point 2: ({x2[0]}, {y2[0]})")

@app.route('/Range-Time-Graph')
def index_7():
    return render_template('task7.html', plot1=fig1.to_json(), plot2=fig2.to_json())


@app.route('/update_7', methods=['POST'])
def update_7():
    global u

    data = request.json
    
    angle_blue = float(data['angle_blue'])
    angle_green = float(data['angle_green'])
    angle_red = float(data['angle_red'])
    angle_yellow = float(data['angle_yellow'])
    u = float(data['speed'])

    theta_blue = np.deg2rad(angle_blue)
    theta_green = np.deg2rad(angle_green)
    theta_red = np.deg2rad(angle_red)
    theta_yellow = np.deg2rad(angle_yellow)

    r_blue_new = calculate_range_7(t, theta_blue)
    r_green_new = calculate_range_7(t, theta_green)
    r_red_new = calculate_range_7(t, theta_red)
    r_yellow_new = calculate_range_7(t, theta_yellow)

    x_blue_new, y_blue_new = calculate_displacements_7(t, theta_blue)
    x_green_new, y_green_new = calculate_displacements_7(t, theta_green)
    x_red_new, y_red_new = calculate_displacements_7(t, theta_red)
    x_yellow_new, y_yellow_new = calculate_displacements_7(t, theta_yellow)

    # Update the corresponding traces based on the angles
    fig1.data[0].y = r_blue_new
    fig1.data[1].y = r_green_new
    fig1.data[2].y = r_red_new
    fig1.data[3].y = r_yellow_new

    fig2.data[0].x = x_blue_new
    fig2.data[0].y = y_blue_new
    fig2.data[1].x = x_green_new
    fig2.data[1].y = y_green_new
    fig2.data[2].x = x_red_new
    fig2.data[2].y = y_red_new
    fig2.data[3].x = x_yellow_new
    fig2.data[3].y = y_yellow_new

    # Clear existing min max points
    fig1.data = [trace for trace in fig1.data if 'Min Max Height' not in trace.name]
    fig2.data = [trace for trace in fig2.data if 'Min Max Height' not in trace.name]

    # Calculate and add new min max points
    minMaxTimes = compute_min_max_times_7(u, g, [angle_blue, angle_green, angle_red, angle_yellow])

    for angle, t1, t2 in minMaxTimes:
        theta = np.deg2rad(angle)
        r1 = calculate_range_7(t1, theta)
        r2 = calculate_range_7(t2, theta)
        x1, y1 = calculate_displacements_7(t1, theta)
        x2, y2 = calculate_displacements_7(t2, theta)
        
        if len(x1) > 0 and len(y1) > 0 and len(x2) > 0 and len(y2) > 0:
            fig1.add_trace(go.Scatter(x=[t1, t2], y=[r1, r2], mode='markers', name=f'Angle = {angle}° Min Max Height', marker=dict(size=10, color='white')))
            fig2.add_trace(go.Scatter(x=[x1[0], x2[0]], y=[y1[0], y2[0]], mode='markers', name=f'Angle = {angle}° Min Max Height', marker=dict(size=10, color='white')))
        
            fig1.update_layout(
                paper_bgcolor='#26252C',  # Background color of the entire paper
                plot_bgcolor='#26252C',   # Background color of the plot area
                font=dict(color='white')  # Font color
            )

            # Update layout for fig2
            fig2.update_layout(
                paper_bgcolor='#26252C',  # Background color of the entire paper
                plot_bgcolor='#26252C',   # Background color of the plot area
                font=dict(color='white')  # Font color
            )

            # Print min max points for X-Y graph
            print(f"Angle = {angle}°, Point 1: ({x1[0]}, {y1[0]}), Point 2: ({x2[0]}, {y2[0]})")

    # Return updated plots JSON
    return jsonify({'plot1': fig1.to_json(), 'plot2': fig2.to_json()})

#Task 8

@app.route('/task8')
def index_8():
    return render_template('task8.html')

@app.route('/update_animated_8', methods=['POST'])

def update_animated_8():

    data = request.json
    bounces = int(data.get('bounces', 5))
    e = float(data.get('e', 0.8))
    initial_speed = float(data.get('initial_speed', 20))
    launch_angle = float(data.get('launch_angle', 45))

    g = 9.81  
    dt = 0.01  


    v0_x = initial_speed * np.cos(np.radians(launch_angle))
    v0_y = initial_speed * np.sin(np.radians(launch_angle))
    x0, y0 = 0, 0


    x = [x0]
    y = [y0]


    vx, vy = v0_x, v0_y
    bounceCount = 0

    while bounceCount < bounces:
        while y[-1] >= 0:
            x_new = x[-1] + vx * dt
            y_new = y[-1] + vy * dt - 0.5 * g * dt**2
            vy -= g * dt
            
            x.append(x_new)
            y.append(y_new)

        #check if contact with floor
        if y[-1] < 0:
            y[-1] = 0  # reset to 0 so it doesnt goes through the floor
            vy = -e * vy  #applying cor in the opposite direction
            bounceCount += 1

    x = np.array(x)
    y = np.array(y)

    #range for plot
    xm = np.min(x) - 1.5
    xM = np.max(x) + 1.5
    ym = np.min(y) - 1.5
    yM = np.max(y) + 1.5

    #total frames of animation
    N_frames = len(x)

    #creating the frames
    frames = [
        go.Frame(
            data=[
                go.Scatter(x=x[:k], y=y[:k],
                           mode="lines",
                           line=dict(width=2, color="blue")),
                go.Scatter(x=[x[k]], y=[y[k]],
                           mode="markers",
                           marker=dict(color="red", size=10))
            ],
            name=f'Frame {k}'
        ) for k in range(N_frames)
    ]

    #plotting
    fig = go.Figure(
        data=[
            go.Scatter(x=x, y=y,
                    mode="lines",
                    line=dict(width=2, color="blue")),
            go.Scatter(x=[x[0]], y=[y[0]],
                    mode="markers",
                    marker=dict(color="red", size=10))
        ],
        layout=go.Layout(
            title='',
            xaxis_title='x (m)',
            yaxis_title='y (m)',
            paper_bgcolor='#26252C',
            plot_bgcolor='#26252C',
            font=dict(color='white'),
            xaxis=dict(range=[xm, xM], autorange=False, zeroline=False),
            yaxis=dict(range=[ym, yM], autorange=False, zeroline=False),
            showlegend=False,
            hovermode="closest",
        ),
        frames=frames
    )

    graphJSON = fig.to_json()
    return jsonify(graphJSON)


def compute_trajectory_9(u, theta_deg, h, m, cd, A):

    g = 9.81 
    rho = 1  #although i researched in that its 1.29 but the document uses 1 so i use 1
    dt = 0.01  # Time step (s)
    theta = np.deg2rad(theta_deg)


    k = 0.5 * cd * rho * A / m
    
    #initial conditions
    vx = u * np.cos(theta)
    vy = u * np.sin(theta)
    
    x = 0
    y = h
    t = 0
    
    xCords = [x]
    yCords = [y]
    
    #verlet
    while y >= 0:
        v = np.sqrt(vx**2 + vy**2)
        
        
        ax = -vx * k * v
        ay = -g - vy * k * v  

        xNew = x + vx * dt + 0.5 * ax * (dt**2)
        yNew = y + vy * dt + 0.5 * ay * (dt**2)
        
        vxNew = vx + ax * dt
        vyNew = vy + ay * dt
        
        x, y = xNew, yNew
        vx, vy = vxNew, vyNew #overwriting old
        t += dt
        
        xCords.append(x)
        yCords.append(y)
    
    #without drag
    xNoDragCords = [0]
    yNoDragCords = [h]
    
    vxNoDrag = u * np.cos(theta)
    vyNoDrag = u * np.sin(theta)
    x = 0
    y = h
    t = 0
    
    while y >= 0:
        x += vxNoDrag * dt
        y += vyNoDrag * dt - 0.5 * g * (dt**2)
        vyNoDrag -= g * dt
        t += dt
        
        xNoDragCords.append(x)
        yNoDragCords.append(y)
    
    
    return xCords, yCords, xNoDragCords, yNoDragCords

@app.route('/page9')
def index_9():
    return render_template('task9.html')

@app.route('/update_9', methods=['POST'])
def update_9():
    u = float(request.json['speed'])
    theta_deg = float(request.json['angle'])
    h = float(request.json['height'])
    m = float(request.json['mass'])
    cd = float(request.json['drag'])
    A = float(request.json['area'])
    
    x_data, y_data, x_data_no_drag, y_data_no_drag = compute_trajectory_9(u, theta_deg, h, m, cd, A)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', name='With air resistance'))
    fig.add_trace(go.Scatter(x=x_data_no_drag, y=y_data_no_drag, mode='lines', name='No air resistance', line=dict(dash='dash')))
    fig.update_layout(title='Projectile Motion with and without Air Resistance',
                      xaxis_title='x /m',
                      yaxis_title='y /m',
                      paper_bgcolor='#26252C', 
                      plot_bgcolor='#26252C',
                      font=dict(color='white'))
    graphJSON = pio.to_json(fig)
    return jsonify(graphJSON)

#Extension

@app.route('/extension')
def index_10():
    return render_template('extension.html')

@app.route('/extensionGlowscript.html')
def extension_glowscript():
    return send_from_directory('templates', 'extensionGlowscript.html')

if __name__ == '__main__':
    app.run(port=5000, host='0.0.0.0')
