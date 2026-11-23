# Optional car model

Without a model the viewer draws its built-in car (`viewer/car.js`, an original crossover at
Model Y dimensions). To show a real car instead, put a glTF model and a `car.json` here, next
to `index.html`:

```
models/
  car.json
  car.glb              (or car/<name>.gltf with its .bin and textures)
```

`otrv car fetch` installs one into `~/.otrv/models/`, and `otrv view` copies that folder here.
Everything in this folder except this README is git-ignored (`viewer/models/*`): models are
usually licensed for personal use only, so they never go into the repository.

## car.json

```json
{
  "file": "car.glb",
  "lengthM": 4.794,
  "paint": "#f2f3f5",
  "credit": "\"2025 Tesla Model Y\" by BloxBloger (https://sketchfab.com/...), CC BY-NC 4.0",
  "source": { "axes": "+Y up, -Z forward" }
}
```

| Field          | Meaning                                                                                  |
|----------------|------------------------------------------------------------------------------------------|
| `file`         | path inside `models/` (`car.glb` or `car/<name>.gltf`); no `..`                          |
| `lengthM`      | overall length in metres; the model is scaled uniformly to it                            |
| `paint`        | body colour; paint materials become a pearl clearcoat in this colour                     |
| `credit`       | attribution, shown small at the bottom-left (hidden with `?ui=0`, exposed as `otrvRender.credit`) |
| `source.axes`  | the model's axes, e.g. `"+Y up, -Z forward"` (also accepted as top-level `axes`). Without it the nose is guessed from where the head lamps sit |

## How the model is used

Matched by material (or mesh) name, case-insensitive:

| Name contains                                   | Treated as                                                         |
|-------------------------------------------------|--------------------------------------------------------------------|
| `exterior_paint`, `paint`, `body color`, `carpaint` | body paint, repainted with `paint`                             |
| `taillight`, `tail light`, `rear lamp`          | tail lamps: brighten when braking                                  |
| `signature`, `headlight`, `head lamp`, `drl`    | front lamps: brighten with the headlights                          |
| `tire` / `tyre`                                 | the four wheel centres (tyre bounding boxes)                       |
| `wheel`, `rim`, `rotor`, `caliper`, `hub`        | re-parented to the nearest wheel, so wheels steer and spin        |

The outer left and right parts of the tail and front lamps (more than 0.42 m from the centre
line) flash amber as blinkers. Loading uses three.js `GLTFLoader` with the meshopt, Draco and
KTX2 decoders, so compressed models work. If anything fails the built-in car is used and the
browser console says why.
