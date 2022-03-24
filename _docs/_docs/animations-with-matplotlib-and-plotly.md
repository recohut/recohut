---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.7
  kernelspec:
    display_name: Python 3
    name: python3
---

```python id="zCGeNykFfBnb" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1636906375785, "user_tz": -330, "elapsed": 3479, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="07fd4485-2850-4bf2-cc07-e18c163272b2"
!pip install gapminder
```

```python id="glOt_IGMhpK3"
%matplotlib inline
```

```python id="98dmiDFefBk9"
from gapminder import gapminder
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib import rc
import seaborn as sns
import numpy as np
import matplotlib

import warnings
warnings.filterwarnings('ignore')

rc('animation', html='html5')
```

```python id="wsK-uFSxOQBL" colab={"base_uri": "https://localhost:8080/", "height": 419} executionInfo={"status": "ok", "timestamp": 1636906378787, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a6f924a9-c964-4fca-a335-8aa6b32c7c2b"
gapminder
```

```python id="J--C1BhzfPpJ"
fig, ax = plt.subplots(figsize=(10, 5))

scatter_data = gapminder.copy()

# Create a color depending on
conditions = [
  scatter_data.continent == 'Asia',
  scatter_data.continent == 'Europe',
  scatter_data.continent == 'Africa',
  scatter_data.continent == 'Americas',
  scatter_data.continent == 'Oceania',
]

values = list(range(5))

scatter_data['color'] = np.select(conditions, values)

font = {
    'weight': 'normal',
    'size'  :  40,
    'color': 'lightgray'
}

years = scatter_data['year'].unique()
data_temp = scatter_data.loc[scatter_data['year'] == years[-1], :]

colors =[f'C{i}' for i in np.arange(1, 6)]
cmap, norm = matplotlib.colors.from_levels_and_colors(np.arange(1, 5+2), colors)

label = ax.text(0.95, 0.25, years[0],
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes,
                fontdict=font)

def update_scatter(i):
    year = years[i]
    data_temp = scatter_data.loc[scatter_data['year'] == year, :]
    ax.clear()
    label = ax.text(0.95, 0.20, years[i],
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes,
                fontdict=font)
    ax.scatter(
        data_temp['gdpPercap'],
        data_temp['lifeExp'],
        s=data_temp['pop']/500000, 
        alpha = 0.5, 
        c=data_temp.color, 
        cmap=cmap,
        norm=norm
    )
    label.set_text(year)

anim = animation.FuncAnimation(fig, update_scatter, frames = len(years), interval = 30, blit=False)
anim
```

```python id="h0ldReG5nOIg"
barchartrace_data  = gapminder.copy()
n_observations = 10
n_frames_between_states = 30

barchartrace_data= barchartrace_data.pivot('year', 'country', 'gdpPercap')
barchartrace_data['year'] = barchartrace_data.index

barchartrace_data.reset_index(drop = True, inplace = True)
barchartrace_data.index = barchartrace_data.index * n_frames_between_states
barchartrace_data =  barchartrace_data.reindex(range(barchartrace_data.index.max()+1))
barchartrace_data = barchartrace_data.interpolate()

# Hacemos otro pivot para volver a los datos originales
barchartrace_data = barchartrace_data.melt(id_vars='year', var_name ='country', value_name  = 'gdpPercap')

import math

n_observations = 10

fig, ax = plt.subplots(figsize=(10, 5))

font = {
    'weight': 'normal',
    'size'  :  40,
    'color': 'lightgray'
}

years = barchartrace_data['year'].unique()

label = ax.text(0.95, 0.20, years[0],
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes,
                fontdict=font)

colors = plt.cm.Dark2(range(200))

def update_barchart_race(i):

    year = years[i]

    data_temp = barchartrace_data.loc[barchartrace_data['year'] == year, :]

    # Create rank and get first 10 countries
    data_temp['ranking'] = data_temp['gdpPercap'].rank(method = 'first',ascending = False)
    data_temp = data_temp.loc[data_temp['ranking'] <= n_observations]

    ax.clear()
    ax.barh(y = data_temp['ranking'] ,
            width = data_temp.gdpPercap, 
            tick_label=data_temp['country'],
           color=colors)

    label = ax.text(0.95, 0.20, math.floor(year),
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes,
                fontdict=font)

    ax.set_ylim(ax.get_ylim()[::-1]) # Revert axis


anim = animation.FuncAnimation(fig, update_barchart_race, frames = len(years))
anim
```

```python id="5zDZ-UPlgpen"
import numpy as np
import seaborn as sns
from scipy.integrate import odeint

# mode parameteres
Ea  = 72750     # activation energy J/gmol
R   = 8.314     # gas constant J/gmol/K
k0  = 7.2e10    # Arrhenius rate constant 1/min
V   = 100.0     # Volume [L]
rho = 1000.0    # Density [g/L]
Cp  = 0.239     # Heat capacity [J/g/K]
dHr = -5.0e4    # Enthalpy of reaction [J/mol]
UA  = 5.0e4     # Heat transfer [J/min/K]
q = 100.0       # Flowrate [L/min]
cAi = 1.0       # Inlet feed concentration [mol/L]
Ti  = 350.0     # Inlet feed temperature [K]
cA0 = 0.5;      # Initial concentration [mol/L]
T0  = 350.0;    # Initial temperature [K]
Tc  = 305.0     # Coolant temperature [K]

# Arrhenius rate expression
def k(T):
    return k0*np.exp(-Ea/R/T)

def deriv(y,t):
    cA,T = y
    dcA = (q/V)*(cAi - cA) - k(T)*cA
    dT = (q/V)*(Ti - T) + (-dHr/rho/Cp)*k(T)*cA + (UA/V/rho/Cp)*(Tc-T)
    return [dcA,dT]

# create a set of initial conditions
ICs = [[cA0,T0] for cA0 in [0] for T0 in np.linspace(295,480,19)]
ICs += [[cA0,T0] for cA0 in np.linspace(0,1,21) for T0 in [290]]
ICs += [[cA0,T0] for cA0 in [1] for T0 in np.linspace(295,475,18)]

# perform simulations for each of the initial conditions
t = np.linspace(0,10.0,800)
sols = [odeint(deriv,IC,t) for IC in ICs]

# create background figure and axes
sns.set(font_scale=1.5)
fig, ax = plt.subplots(figsize=(8,8))
ax.set_xlim((0,1))
ax.set_ylim((290,480))
ax.set_xlabel('Concentration [gmol/liter]')
ax.set_ylabel('Temperature [K]')
ax.set_title('Exothermic Reactor with Tc = {0:.1f} K'.format(Tc))

# create lists of colors, points, and lines
colors = sns.color_palette("husl",len(sols))
pts = sum([ax.plot([],[],'o',color=colors[k],ms=15) for k in range(0,len(sols))],[])
lines = sum([ax.plot([],[],color=colors[k],lw=2) for k in range(0,len(sols))],[])

# don't show the plain background
plt.close()

# define function to draw each frame
def drawframe(n):
    for k in range(0,len(sols)):
        C,T = sols[k].T
        pts[k].set_data(C[n],T[n])
        lines[k].set_data(C[:n],T[:n])
    return pts + lines

# create animiation object and render in HTML video
anim = animation.FuncAnimation(fig, drawframe, frames=len(t), interval=20, blit=True)
anim
```

```python colab={"base_uri": "https://localhost:8080/", "height": 542} id="5k02uf8Ri0O_" executionInfo={"status": "ok", "timestamp": 1636880489523, "user_tz": -330, "elapsed": 4426, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="74e80284-8d92-469a-ae7e-7edee0388766"
import plotly.express as px
df = px.data.gapminder()
px.scatter(df, x="gdpPercap", y="lifeExp", animation_frame="year", animation_group="country",
           size="pop", color="continent", hover_name="country",
           log_x=True, size_max=55, range_x=[100,100000], range_y=[25,90])
```

```python id="ndq_Z9bmog2J"
import pandas as pd

!wget -q --show-progress http://ucdp.uu.se/downloads/ged/ged172-csv.zip
!unzip ged172-csv.zip

df = pd.read_csv('ged171.csv')
df.head()
```

```python id="D5vdgUESon35"
df.sort_values(by='year', inplace=True)
df.info()
```

```python id="x2B7kZCHoqv0"
fig = px.bar(df, x="region", y="high", color="region",
  animation_frame="year", animation_group="country", range_y=[0,10000])
fig.show()
```
