# Pathgen

A project for generating svg generative art. To be used for plotters.

Currently in experimental phase.

## Running

```bash
git clone git@github.com:angelocarly/pathgen.git
cd pathgen
cargo run
```

## Viewing the path

I use [vpype](https://github.com/abey79/vpype) to view the path.

```bash
vpype read path.svg show
```

## Exporting to gcode

SVG can be exported to gcode, using the [vpype-gcode](https://github.com/plottertools/vpype-gcode) plugin.

```bash
vpype read path.svg gwrite out.gcode
```