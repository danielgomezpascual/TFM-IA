# Tipos de crisis (corpus TUSZ)

Cada evento corresponde a una fila del `.csv_bi`. El tipo es la etiqueta multiclase dominante. Las definiciones proceden del fichero `seizures_types_v02.xlsx` (30 clases de anotación de TUSZ, códigos 0 a 29); la descripción está traducida al español.

## Clases de crisis definidas en el modelo (12)

| Código | Nº | Nombre | Signos | Localización | Descripción |
|---|---:|---|---|---|---|
| **SEIZ** | 7 | Seizure | Clínico y electrográfico | Todas | Clase común de crisis que puede incluir todos los tipos |
| **FNSZ** | 8 | Focal Non-Specific Seizure | Electrográfico | Hemisférica/Focal | Crisis focales que no pueden especificarse según su tipo |
| **GNSZ** | 9 | Generalized Non-Specific Seizure | Electrográfico | Generalizada | Crisis generalizadas que no pueden clasificarse en ninguno de los grupos siguientes |
| **SPSZ** | 10 | Simple Partial Seizure | Clínico y electrográfico | Todas | Crisis parciales con la conciencia conservada; el tipo se especifica solo por los signos clínicos |
| **CPSZ** | 11 | Complex Partial Seizure | Clínico y electrográfico | Todas | Crisis parciales con pérdida de conciencia; el tipo se especifica solo por los signos clínicos |
| **ABSZ** | 12 | Absence Seizure | Clínico y electrográfico | Generalizada | Descargas de ausencia observadas en el EEG; el paciente pierde la conciencia unos segundos (pequeño mal) |
| **TNSZ** | 13 | Tonic Seizure | Clínico y electrográfico | Todas | Rigidez del cuerpo durante la crisis (el efecto en el EEG desaparece) |
| **CNSZ** | 14 | Clonic Seizure | Clínico y electrográfico | Todas | Sacudidas/temblores del cuerpo durante la crisis |
| **TCSZ** | 15 | Tonic Clonic Seizure | Clínico y electrográfico | Todas | Primero rigidez y después sacudidas del cuerpo (gran mal) |
| **ATSZ** | 16 | Atonic Seizure | Clínico y electrográfico | N/A | Pérdida súbita del tono muscular |
| **MYSZ** | 17 | Myoclonic Seizure | Clínico y electrográfico | N/A | Sacudidas mioclónicas de las extremidades |
| **NESZ** | 18 | Non-Epileptic Seizure | Clínico y electrográfico | N/A | Cualquier crisis no epiléptica observada. No contiene signos electrográficos |

## Distribución por partición

`ev` = número de eventos, `pac` = número de pacientes, `min` = minutos totales de crisis.

| tipo | train_ev | train_pac | train_min | dev_ev | dev_pac | dev_min | eval_ev | eval_pac | eval_min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **absz** · Ausencia | 47 | 8 | 7.4 | 24 | 4 | 5.3 | 49 | 1 | 4.4 |
| **cpsz** · Focal con alteración de conciencia (parcial compleja) | 258 | 31 | 436.7 | 3 | 2 | 1.2 | 41 | 3 | 52.7 |
| **fnsz** · Focal no especificada | 1139 | 101 | 1121.7 | 263 | 22 | 238.4 | 153 | 17 | 196.7 |
| **gnsz** · Generalizada no especificada | 311 | 48 | 352.7 | 199 | 25 | 195.7 | 108 | 12 | 134.0 |
| **mysz** · Mioclónica | 2 | 1 | 21.5 | 0 | 0 | 0.0 | 0 | 0 | 0.0 |
| **sin_tipo** · Sin tipo asignado (probablemente SEIZ, clase 7) | 1 | 1 | 0.0 | 0 | 0 | 0.0 | 0 | 0 | 0.0 |
| **spsz** · Focal con conciencia conservada (parcial simple) | 8 | 1 | 10.2 | 0 | 0 | 0.0 | 0 | 0 | 0.0 |
| **tcsz** · Tónico-clónica | 31 | 9 | 34.4 | 2 | 1 | 3.3 | 8 | 4 | 10.1 |
| **tnsz** · Tónica | 15 | 2 | 5.8 | 1 | 1 | 2.4 | 1 | 1 | 0.7 |

Clases de crisis del fichero que no aparecen en esta tabla: **cnsz** (clónica), **atsz** (atónica) y **nesz** (no epiléptica).

## Duración media por evento (train)

| tipo | min/evento | segundos/evento |
|---|---:|---:|
| **sin_tipo** · Sin tipo asignado (probablemente SEIZ, clase 7) | 0.00 | 0.0 |
| **absz** · Ausencia | 0.16 | 9.4 |
| **tnsz** · Tónica | 0.39 | 23.2 |
| **fnsz** · Focal no especificada | 0.98 | 59.1 |
| **tcsz** · Tónico-clónica | 1.11 | 66.6 |
| **gnsz** · Generalizada no especificada | 1.13 | 68.0 |
| **spsz** · Focal con conciencia conservada (parcial simple) | 1.27 | 76.5 |
| **cpsz** · Focal con alteración de conciencia (parcial compleja) | 1.69 | 101.6 |
| **mysz** · Mioclónica | 10.75 | 645.0 |

## Observaciones

- **Desbalance fuerte**: fnsz domina (unos 1.100 eventos en train); gnsz y cpsz le siguen a distancia.
- **Cobertura de clases**: El fichero define 12 clases de crisis (SEIZ, FNSZ, GNSZ, SPSZ, CPSZ, ABSZ, TNSZ, CNSZ, TCSZ, ATSZ, MYSZ, NESZ), pero en la tabla de eventos solo aparecen 9 etiquetas. No aparecen cnsz (clónica), atsz (atónica) ni nesz (no epiléptica), y sin_tipo es muy probablemente SEIZ (clase 7). Conviene confirmarlo en el script de conversión.
- **Clases inutilizables para evaluar**: mysz, spsz y sin_tipo solo aparecen en train, con 1 paciente cada una, y no tienen ejemplos en dev ni en eval. Solo cabe fusionarlas o descartarlas.
- **Métricas frágiles**: absz tiene 1 solo paciente en eval; tnsz tiene 1 paciente en dev y 1 en eval; cpsz en dev tiene 3 eventos (2 pacientes). Cualquier métrica por clase ahí depende de muy pocos pacientes.
- **Distribución desigual entre particiones**: cpsz pasa de 258 eventos en train a 3 en dev, y absz de 8 pacientes en train a 1 en eval. Conviene mirar los pacientes además de los eventos.
- **Duración por tipo**: cpsz dura de media unos 1,7 min por evento; absz y tnsz son mucho más breves (unos 9 s y 23 s). Esto afecta a cómo se definen las ventanas.
- **Ojo con mysz**: 2 eventos suman 21,5 min (unos 10,7 min por evento), muy por encima del resto. Puede ser un artefacto de anotación.
- **tnsz y el EEG**: Según el fichero, en la crisis tónica "el efecto en el EEG desaparece". Para un detector basado en EEG puede ser una clase especialmente difícil.
- **nesz no es detectable por EEG**: El fichero indica que no contiene signos electrográficos, así que, si llegara a aparecer, no tendría sentido como clase de un modelo que solo ve EEG.
- **Localización según el fichero**: Solo fnsz (Hemisférica/Focal), gnsz y absz (Generalizada) tienen localización definida. seiz, spsz, cpsz, tnsz, cnsz y tcsz figuran como "Todas", y atsz, mysz y nesz como "N/A". Hay que tenerlo en cuenta si se agrupa en focal vs. generalizada.

## Anexo: resto de clases del modelo (no son crisis)

| Código | Nº | Nombre | Signos | Localización | Descripción |
|---|---:|---|---|---|---|
| **NULL** | 0 | No Event | N/A | N/A | Evento sin clasificar |
| **SPSW** | 1 | Spike/Sharp and Wave | Electrográfico | Todas | Complejos de punta y onda, complejos de onda aguda y onda |
| **GPED** | 2 | Generalized Periodic Epileptiform Discharges | Electrográfico | Generalizada | Descargas periódicas difusas |
| **PLED** | 3 | Periodic Lateralized Epileptiform Discharges | Electrográfico | Hemisférica/Focal | Descargas periódicas focales |
| **EYBL** | 4 | Eye blink | Clínico y electrográfico | Focal | Tipo específico de artefacto de movimiento ocular, agudo y de alta amplitud, correspondiente a parpadeos |
| **ARTF** | 5 | Artifacts (All) | Clínico y electrográfico | Todas | Cualquier señal eléctrica que no procede del cerebro, como las debidas al equipo o a factores ambientales |
| **BCKG** | 6 | Background | Electrográfico | N/A | Eventos basales / sin interés |
| **INTR** | 19 | Interesting Patterns | Electrográfico | Todas | Cualquier patrón inusual o interesante observado que no encaja en las clases anteriores |
| **SLOW** | 20 | Slowing | Electrográfico | Todas | Disminución breve de la frecuencia |
| **EYEM** | 21 | Eye Movement Artifact | Clínico y electrográfico | Focal | Artefacto frontal/prefrontal muy común que aparece al mover los ojos |
| **CHEW** | 22 | Chewing Artifact | Clínico y electrográfico | Todas | Artefacto específico que afecta a varios canales y corresponde a la masticación del paciente, en ráfagas |
| **SHIV** | 23 | Shivering Artifact | Clínico y electrográfico | Todas | Artefacto específico, agudo y sostenido, que corresponde a temblores (escalofríos) del paciente |
| **MUSC** | 24 | Muscle Artifact | Clínico y electrográfico | Todas | Artefacto muy común, agudo y de alta frecuencia, que corresponde a agitación/nerviosismo del paciente |
| **ELPP** | 25 | Electrode Pop Artifact | Clínico y electrográfico | Focal | Artefacto breve caracterizado por canales que usan el mismo electrodo con picos de simetría perfecta |
| **ELST** | 26 | Electrostatic Artifact | Clínico y electrográfico | Todas | Artefacto causado por movimiento o interferencias en los electrodos, con morfologías variadas |
| **CALB** | 27 | Calibration Artifact | Clínico y electrográfico | Todas | Artefacto causado por la calibración de los electrodos. Aparece como un aplanamiento de la señal al inicio de los archivos |
| **HPHS** | 28 | Hypnagogic Hypersynchrony | Electrográfico | Generalizada | Breve periodo de ondas lentas de alta amplitud |
| **TRIP** | 29 | Triphasic Wave | Electrográfico | Generalizada | Ondas grandes de tres fases, frecuentemente causadas por una condición metabólica subyacente |
