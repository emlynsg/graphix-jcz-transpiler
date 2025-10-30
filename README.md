# Graphix Transpiler from Quantum Circuit to MBQC Patterns via J-∧z decomposition

This package provides a transpiler from quantum circuits to MBQC
(Measurement-Based Quantum Computing) patterns via J-∧z decomposition,
designed for use with the [Graphix library](https://github/TeamGraphix/graphix).

In the seminal paper [*The Measurement
Calculus*](https://arxiv.org/abs/0704.1263) by Danos, Kashefi, and
Panangaden (2007), circuit-to-pattern transpilation leverages the
universality of the gate set consisting of 𝔍(α) and ∧Z. This package
implements that transpilation mehtod in a straightforward and
principled way.

Compared to the existing transpilation procedure in Graphix, this
implementation is more naive but also more transparent:

- The code closely follows the structure described in the literature,
  making it easier to follow and verify for correctness.

- The resulting MBQC patterns are simple to understand and predict.

- Further optimization is possible via standard techniques such as
  Pauli pre-simulation, and space or depth minimization.

Todo:
 - [ ] Correct functionality with CCX gate
 - [ ] Set up for direct and OG/flow generation

Functionality relative to the rest of Graphix: 

```mermaid
---
config:
  layout: dagre
---
flowchart TB
 subgraph s1["via Flow"]
        n10("**PauliFlow**")
        n11("**GFlow**")
        n12("**CausalFlow**")
  end
 subgraph s2["current transpiler"]
        n21("**Circuit**")
        n22("**TranspileResult**")
        n23("transpile with handwritten conversions")
  end
 subgraph n31["JCZ Instructions"]
        n313["J"]
        n314["CZ"]
  end
 subgraph s3["JCZ transpiler"]
        n31
        n32("JCZ decomposition functions")
        n33("transpile using conversions in M.C.")
  end
 subgraph s4["Brickwork transpiler"]
        n40["construct bricks using conversions in UBQC (currently only CNOT, RX, RZ)"]
        n41["**Brick**"]
        n42["**Layer**"]
        n43["*Measurement table*"]
        n44["converts using brickwork structure"]
        n45["add other bricks in UBQC (I, T, H)"]
  end
 subgraph tr1["Transpilers"]
        s2
        s3
        s4
  end
    n10 -.-> n11
    n11 -.-> n12
    n21 --> n23 & n32 & n40
    n23 --> n22
    n0("**OpenGraph**") --> s1 & n3("**Pattern**")
    s1 --> n0 & n2("**XZCorrections**")
    n2 -- add total order --> n3
    n3 --> n2 & n0
    n2 --> n0
    n22 --> n3
    n32 --> n31
    n31 --> n33
    n31 ==> n41
    n33 --> n22
    n40 ==> n32
    n40 --> n41
    n41 --> n42
    n42 --> n43
    n43 --> n44
    n44 --> n3
    n45 ==> n40
    n33 == J for measurements and CZ for OG ==> n0
    n44 ==> n31
    style n45 fill:#FFCDD2
    style s2 fill:#FFE0B2
    style s3 fill:#E1BEE7
    style s4 fill:#E1BEE7
    style tr1 fill:#C8E6C9
    linkStyle 2 stroke:#000000,fill:none
    linkStyle 5 stroke:#000000,fill:none
    linkStyle 9 stroke:#000000,fill:none
    linkStyle 10 stroke:#000000,fill:none
    linkStyle 17 stroke:#D50000,fill:none
    linkStyle 19 stroke:#D50000,fill:none
    linkStyle 22 stroke:#000000,fill:none
    linkStyle 23 stroke:#000000,fill:none
    linkStyle 24 stroke:#000000,fill:none
    linkStyle 25 stroke:#D50000,fill:none
    linkStyle 26 stroke:#D50000,fill:none
    linkStyle 27 stroke:#D50000,fill:none
```