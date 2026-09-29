#!/usr/bin/env python3
"""Generate the independent Re=100, 2-D stationary-cylinder OpenFOAM cases.

Only Python's standard library is used.  The two meshes share one structured
background; an annular four-block O-grid replaces each occupied centre box.
The smoke profile preserves the initial 0.5-time-unit trials. The baseline
profile creates separate, perturbed long-run inputs. Both are coarse meshes;
neither is a grid-converged research result.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
HEADER = """FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object OBJECT;
}
"""
X = (0.0, 9.0, 11.0, 14.0, 16.0, 30.0)
Y = (0.0, 6.5, 8.5, 15.0)
NX = (45, 24, 30, 24, 70)
NY = (33, 24, 33)
THICKNESS = 0.1


def write_file(path: Path, body: str, field_class: str = "dictionary") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = HEADER.replace("OBJECT", path.name).replace("class dictionary;", f"class {field_class};")
    path.write_text(header + body, encoding="utf-8")


def mesh_dict(tandem: bool, refinement: int = 1) -> str:
    verts: list[tuple[float, float, float]] = []
    vertex_ids: dict[tuple[float, float, float], int] = {}
    blocks: list[tuple[tuple[int, ...], int, int]] = []
    arcs: list[tuple[int, int, tuple[float, float, float]]] = []
    edges: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}

    def vertex(x: float, y: float, z: float) -> int:
        key = (round(x, 12), round(y, 12), round(z, 12))
        if key not in vertex_ids:
            vertex_ids[key] = len(verts)
            verts.append(key)
        return vertex_ids[key]

    def quad(points: list[tuple[float, float]], n0: int, n1: int) -> None:
        assert len(points) == 4
        area = sum(points[i][0] * points[(i + 1) % 4][1]
                   - points[(i + 1) % 4][0] * points[i][1]
                   for i in range(4))
        assert area > 0, points
        ids = tuple(vertex(x, y, 0.0) for x, y in points)
        blocks.append((ids, n0, n1))
        for i in range(4):
            a, b = ids[i], ids[(i + 1) % 4]
            edges.setdefault(tuple(sorted((a, b))), []).append(
                (a, b, vertex(*verts[a][:2], THICKNESS),
                 vertex(*verts[b][:2], THICKNESS))
            )

    occupied = {(1, 1)} | ({(3, 1)} if tandem else set())
    for ix in range(len(NX)):
        for iy in range(len(NY)):
            if (ix, iy) in occupied:
                continue
            x0, x1, y0, y1 = X[ix], X[ix + 1], Y[iy], Y[iy + 1]
            quad([(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                 NX[ix] * refinement, NY[iy] * refinement)

    for cx in ((10.0, 15.0) if tandem else (10.0,)):
        cy, q = 7.5, 0.5 / math.sqrt(2)
        ilb, irb = (cx - q, cy - q), (cx + q, cy - q)
        ilt, irt = (cx - q, cy + q), (cx + q, cy + q)
        olb, orb = (cx - 1, cy - 1), (cx + 1, cy - 1)
        olt, ort = (cx - 1, cy + 1), (cx + 1, cy + 1)
        ring = (
            ([olb, orb, irb, ilb], 24, 16, ilb, irb, (cx, cy - 0.5)),
            ([irb, orb, ort, irt], 16, 24, irb, irt, (cx + 0.5, cy)),
            ([ilt, irt, ort, olt], 24, 16, ilt, irt, (cx, cy + 0.5)),
            ([olb, ilb, ilt, olt], 16, 24, ilb, ilt, (cx - 0.5, cy)),
        )
        for points, n0, n1, start, end, midpoint in ring:
            quad(points, n0 * refinement, n1 * refinement)
            for z in (0.0, THICKNESS):
                arcs.append((vertex(*start, z), vertex(*end, z), (*midpoint, z)))

    patches: dict[str, list[tuple[int, ...]]] = {
        "inlet": [], "outlet": [], "upperLower": [], "frontCylinder": [],
        "rearCylinder": [], "frontBack": [],
    }
    for members in edges.values():
        if len(members) == 2:
            continue
        assert len(members) == 1, members
        a, b, a_top, b_top = members[0]
        xa, ya, _ = verts[a]
        xb, yb, _ = verts[b]
        if xa == xb == 0.0:
            name = "inlet"
        elif xa == xb == 30.0:
            name = "outlet"
        elif ya == yb == 0.0 or ya == yb == 15.0:
            name = "upperLower"
        else:
            mx = (xa + xb) / 2
            name = "frontCylinder" if mx < 12.5 else "rearCylinder"
        patches[name].append((a, b, b_top, a_top))
    for ids, _, _ in blocks:
        bottom = tuple(reversed(ids))
        top = tuple(vertex(*verts[v][:2], THICKNESS) for v in ids)
        patches["frontBack"].extend((bottom, top))

    lines = ["\nscale 1;\n", "vertices\n("]
    lines.extend(f"    ({x:g} {y:g} {z:g})" for x, y, z in verts)
    lines += [");", "blocks\n("]
    for ids, n0, n1 in blocks:
        top = tuple(vertex(*verts[v][:2], THICKNESS) for v in ids)
        lines.append("    hex (" + " ".join(map(str, ids + top))
                     + f") ({n0} {n1} 1) simpleGrading (1 1 1)")
    lines += [");", "edges\n("]
    lines.extend(f"    arc {a} {b} ({p[0]:.12g} {p[1]:.12g} {p[2]:g})"
                 for a, b, p in arcs)
    lines += [");", "boundary\n("]
    for name, faces in patches.items():
        if name == "rearCylinder" and not tandem:
            continue
        kind = "empty" if name == "frontBack" else "wall" if "Cylinder" in name else "patch"
        lines += [f"    {name}", "    {", f"        type {kind};", "        faces", "        ("]
        lines.extend("            (" + " ".join(map(str, face)) + ")" for face in faces)
        lines += ["        );", "    }"]
    lines += [");", "mergePatchPairs ();", ""]
    return "\n".join(lines)


def write_case(name: str, tandem: bool, baseline: bool = False,
               refinement: int = 1, delta_t: float = 0.01,
               time_scheme: str = "Euler") -> None:
    case = ROOT / "cases" / name
    write_file(case / "system/blockMeshDict", mesh_dict(tandem, refinement))
    write_file(case / "constant/transportProperties", """
transportModel Newtonian;
nu [0 2 -1 0 0 0 0] 0.01;
""")
    write_file(case / "constant/turbulenceProperties", """
simulationType laminar;
""")
    if baseline:
        # A reproducible 2% cross-stream seed behind the front cylinder.
        # Only the initial condition changes; it is not physical actuation.
        write_file(case / "system/setFieldsDict", """
defaultFieldValues
(
    volVectorFieldValue U (1 0 0)
);
regions
(
    boxToCell
    {
        box (10.6 7.55 -0.01) (11.8 8.25 0.11);
        fieldValues
        (
            volVectorFieldValue U (1 0.02 0)
        );
    }
);
""")
    walls = "frontCylinder rearCylinder" if tandem else "frontCylinder"
    write_file(case / "0/U", f"""
dimensions [0 1 -1 0 0 0 0];
internalField uniform (1 0 0);
boundaryField
{{
    inlet {{ type fixedValue; value uniform (1 0 0); }}
    outlet {{ type zeroGradient; }}
    upperLower {{ type slip; }}
    {walls.replace(' ', ' { type noSlip; }\n    ')} {{ type noSlip; }}
    frontBack {{ type empty; }}
}}
""", "volVectorField")
    write_file(case / "0/p", f"""
dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{{
    inlet {{ type zeroGradient; }}
    outlet {{ type fixedValue; value uniform 0; }}
    upperLower {{ type zeroGradient; }}
    {walls.replace(' ', ' { type zeroGradient; }\n    ')} {{ type zeroGradient; }}
    frontBack {{ type empty; }}
}}
""", "volScalarField")
    write_file(case / "system/fvSchemes", f"""
ddtSchemes {{ default {time_scheme}; }}
gradSchemes {{ default Gauss linear; }}
divSchemes
{{
    default none;
    div(phi,U) Gauss linearUpwind grad(U);
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}}
laplacianSchemes {{ default Gauss linear corrected; }}
interpolationSchemes {{ default linear; }}
snGradSchemes {{ default corrected; }}
""")
    write_file(case / "system/fvSolution", """
solvers
{
    p { solver GAMG; tolerance 1e-7; relTol 0.01; smoother GaussSeidel; }
    pFinal { $p; relTol 0; }
    U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.1; }
    UFinal { $U; relTol 0; }
}
PIMPLE
{
    momentumPredictor yes;
    nOuterCorrectors 1;
    nCorrectors 2;
    nNonOrthogonalCorrectors 1;
}
""")
    force_objects = ""
    for suffix, patch, origin in (("Front", "frontCylinder", 10),
                                  ("Rear", "rearCylinder", 15)):
        if patch == "rearCylinder" and not tandem:
            continue
        force_objects += f"""
    force{suffix}
    {{
        type forceCoeffs;
        libs (\"libforces.so\");
        patches ({patch});
        rho rhoInf;
        rhoInf 1;
        CofR ({origin} 7.5 0);
        liftDir (0 1 0);
        dragDir (1 0 0);
        pitchAxis (0 0 1);
        magUInf 1;
        lRef 1;
        Aref {THICKNESS};
        writeControl timeStep;
        writeInterval 1;
    }}
"""
    if tandem:
        # Paper: 32 equally spaced velocity measurements over a 3D vertical
        # segment, 2D downstream of the rear-cylinder centre. Endpoint
        # inclusion here is our explicit interpretation of its schematic.
        locations = "\n".join(
            f"            (17 {6 + 3 * i / 31:.12g} 0.05)"
            for i in range(32)
        )
        force_objects += f"""
    wakeProbes
    {{
        type probes;
        libs (\"libsampling.so\");
        fields (U);
        interpolationScheme cellPoint;
        includeOutOfBounds false;
        writeControl timeStep;
        writeInterval 1;
        probeLocations
        (
{locations}
        );
    }}
"""
    write_file(case / "system/controlDict", f"""
application pimpleFoam;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime {160 if baseline else 0.5};
deltaT {delta_t:g};
writeControl runTime;
writeInterval {2 if baseline else 0.25};
purgeWrite 0;
writeFormat ascii;
writePrecision 9;
runTimeModifiable no;
functions
{{{force_objects}
}}
""")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("single", "tandem", "both"), default="both")
    parser.add_argument("--profile", choices=("smoke", "baseline", "temporal", "medium", "backward", "medium_backward"), default="smoke")
    args = parser.parse_args()
    if args.profile in ("temporal", "medium", "backward", "medium_backward") and args.case != "tandem":
        parser.error("temporal, medium, backward and medium_backward profiles require --case tandem")
    baseline = args.profile != "smoke"
    refinement = 2 if args.profile in ("medium", "medium_backward") else 1
    delta_t = 0.005 if args.profile in ("temporal", "medium", "backward", "medium_backward") else 0.01
    time_scheme = "backward" if args.profile in ("backward", "medium_backward") else "Euler"
    targets = []
    if args.case in ("single", "both"):
        targets.append(("single_baseline" if baseline else "single_static", False))
    if args.case in ("tandem", "both"):
        names = {"smoke": "tandem_static", "baseline": "tandem_baseline",
                 "temporal": "tandem_dt005", "medium": "tandem_medium_dt005",
                 "backward": "tandem_backward_dt005",
                 "medium_backward": "tandem_medium_backward_dt005"}
        targets.append((names[args.profile], True))
    existing = [name for name, _ in targets if (ROOT / "cases" / name).exists()]
    if existing:
        parser.error(f"refusing to overwrite existing case(s): {', '.join(existing)}")
    for name, tandem in targets:
        write_case(name, tandem, baseline, refinement, delta_t, time_scheme)


if __name__ == "__main__":
    main()
