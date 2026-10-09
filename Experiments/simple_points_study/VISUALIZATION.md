# Visualización de la geometría

Desde la raíz del repositorio, con el entorno virtual existente:

```powershell
.\.venv\Scripts\python.exe Experiments\simple_points_study\visualize_simulation.py
```

Para abrir el visor y ejecutar también la simulación original:

```powershell
.\.venv\Scripts\python.exe Experiments\simple_points_study\visualize_simulation.py --simulate
```

También puedes ejecutar `python visualize_simulation.py` desde esta carpeta si
ya tienes el entorno activado. Los resultados se escriben en el directorio desde
el que ejecutas el comando, igual que con `simulate.py`. Se conserva su protección
contra sobrescribir resultados existentes. El progreso y los errores aparecen en
la terminal; cerrar el visor no cancela la simulación.

Arrastra el selector inferior o haz clic en un marcador de ping para elegirlo.
Puedes rotar las vistas con el ratón y usar las herramientas de la ventana para
ampliarlas. La vista izquierda muestra trayectoria, pings, velocidad y targets;
la derecha muestra las posiciones y orientaciones del proyector TX y receptores RX.
Ambas usan metros y Z positivo hacia abajo. La flecha de velocidad tiene una escala
de 1 metro gráfico por 1 m/s; las flechas de orientación miden 0,8 m.

Solo se dibujan aperturas y patrones rectangulares activos. La superficie del haz
representa la magnitud normalizada de la respuesta del plugin, con una escala
gráfica de 1,5 m por unidad de ganancia; no representa el alcance del sonar ni una
onda propagándose. Si `frequency="all"`, se muestra una frecuencia a la vez y,
cuando el intervalo de frecuencias de la señal lo permite, aparece un selector.
El modo `centre`, `min` o `max` usa la frecuencia correspondiente. La esfera se
dibuja cuando está activo `RigidSphereFormFunction`, usando su radio real.

El visor captura en memoria la configuración construida por `simulate.py`,
sustituyendo temporalmente la creación del cluster, el conversor de resultados
y la llamada que calcula los ecos. Restaura estas funciones al terminar la captura.
Con `--simulate`, inicia `simulate.py local` en otro proceso sin esas sustituciones.
Los archivos existentes no se modifican para conectar el visor.

Para exportar una vista sin abrir ventana ni simular:

```powershell
python visualize_simulation.py --save scene.png
```

Requiere Matplotlib, disponible en el entorno virtual actual. Este visor está
dirigido al estudio `simple_points_study` y su función `simulate(cluster)`.
Para colecciones grandes se muestran hasta 10 000 puntos por colección y hasta
100 esferas; todos los pings y receptores configurados se mantienen.

## TX coverage volume

When TX has an active rectangular beampattern, Overview shows an orange,
transparent volume bounded by the forward principal lobe's exact -3 dB contour.
It follows TX's position and orientation at the selected ping. The boundary uses
`20*log10(abs(B)) = -3`, evaluated with the existing plugin's amplitude response;
it excludes the rear lobe and secondary lobes. Horizontal and vertical full
beamwidths are measured through the central cuts in the transducer's local frame.

`TX visual range (m)` controls the maximum radial distance from TX, not an acoustic
detection range. Its initial value extends beyond the displayed targets. The
frequency is the one displayed for the active TX pattern: nominal frequency for
the sine burst; the selected frequency for `all` with a broadband signal, or the
configured `min`, `max`, or `centre` frequency. Changing these display controls
does not change the simulation configuration.

The coverage readout estimates the fraction of **Target 1's sphere surface**
inside that angular/range volume using 4096 equal-area samples. Inside/outside
labels are sampling estimates, so very small intersections can be missed. This
is geometric surface coverage, not visible surface area, transmitted energy,
echo strength, or a sphere-intersection calculation used by the point-target
simulator. For targets without an active rigid-sphere model, the readout counts
displayed target points inside the volume. Outside the volume does not imply no
echo. If multiple rectangular TX distortions are active, this preview uses the
first one rather than their combined response.
