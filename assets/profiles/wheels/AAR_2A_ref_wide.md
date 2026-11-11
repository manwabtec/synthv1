# AAR_2A_ref_wide.csv

- **Source:** supplied by the project owner as "AAR_2A_Ref_Profile_wide 2.csv" (2026-11-10).
  TODO (owner): document/drawing reference, revision and date of the original.
- **Content:** tread + flange contour only (open curve), 1262 points, about 0.12 mm spacing.
  The rim, plate and hub are completed with placeholder dimensions (`WheelSection`).
- **Columns (inferred, please confirm):**
  1. lateral position in mm, 0 at the flange back face, + towards the field side
     (field-side rim face at 145.3 mm);
  2. radial offset in mm from the taping-line radius, + outward (flange tip +27.75 mm;
     the curve crosses 0 near 70 mm, i.e. 2-3/4 in from the flange back).
- **Order:** the points from ~131 mm to the field-side end (145.3 mm) are stored at the end
  of the file; `load_wheel_contour` chains all points into one continuous curve.
