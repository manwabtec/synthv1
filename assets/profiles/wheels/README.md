# Wheel profiles

Put one CSV per wheel profile here. A run's scene config picks one by its file name
without `.csv` (e.g. `wheel_profile: my_wheel_a`).

**Drop the file in as-is**: the loader is written to match the format of the first
real file (columns, units, coordinate convention). Next to each CSV, add a short
`<name>.md` (or header comment lines starting with `#`) recording:

- source (measurement device or drawing, document/revision) and date;
- units (mm or m) and which columns are which;
- coordinate convention: where lateral 0 is (e.g. flange back face or taping line),
  which direction is positive, and what the vertical axis is measured from;
- whether it is the tread + flange contour only, or the full wheel cross-section.

If only the tread + flange contour is given, the rim, plate and hub are completed with
clearly labelled placeholder geometry.
