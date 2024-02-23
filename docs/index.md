# IHSetDean

## Summary

Miller and Dean (2004) proposed a simple model for shoreline evolution using several field datasets. The model is developed based on the observation that shoreline positions change as a function of an equilibrium position. The model includes three adjustable parameters that represent the baseline conditions under which shoreline displacement is calculated to minimize the error. This model is very efficient because it only represents the shoreline response to the process and only requires input of readily available storm surge and water level data.

## Model formula

Miller and Dean (2004) suggested a simple shoreline evolution model based on the imbalance of shoreline change between an equilibrium shoreline change and shoreline position as follows:

```text
(∂S(t))/∂t=k(S_eq (t)-S(t))

S(t) : the shoreline position at time t
S_eq : the equilibrium shoreline position
k : the calibration parameter for the rate at which the shoreline approaches equilibrium (k; k=k_a H_b^2; k=k_a H_b^3; k=k_a Ω)
```

Miller and Dean (2004) proposed an equilibrium shoreline change owing to the change of sea level (Fig. 4 1):

```text
S_eq=-W^* (t)((0.068H_b+S)/(B+1.28H_b ))

H_b : the breaking wave height
S : the change in local water level
B : the berm wave height
W^* : the width of the active surf zone
```

![Definition sketch of shoreline evolution](_static/images/Imagen1.png)

Fig. 4 1. Definition sketch of shoreline evolution according the change of water level owing to storm surge and wave setup (Miller and Dean, 2004).