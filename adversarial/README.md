# Adversarial Robustness of Micro-Doppler Classifiers

**Research question:** Can a motivated adversary defeat a state-of-the-art counter-UAV classifier with cheap physical modifications to the drone?

**Working hypothesis:** Yes. Every published micro-Doppler classifier (including the baseline in `/baseline`) assumes drones behave like drones — rotating propellers at predictable RPMs, standard blade counts, typical flight trajectories. An attacker who violates any of these assumptions breaks the classifier without breaking the drone.

---

## Threat taxonomy (planned)

| ID  | Attack                                       | Est. cost     | Mechanism                                                                 |
|-----|----------------------------------------------|---------------|---------------------------------------------------------------------------|
| A1  | Lower propeller RPM                          | ₹0            | Shifts blade-flash frequency toward bird range                            |
| A2  | Fewer blades per propeller (1-blade)         | ₹500          | Halves BFP frequency; combined with A1 pushes into bird range             |
| A3  | Dielectric-coated propellers                 | ₹100          | Reduces blade-flash amplitude below detection threshold                   |
| A5  | Contra-rotating coaxial propellers           | ₹5,000        | Net micro-Doppler cancels                                                 |
| A6  | Ducted-fan airframe                          | ₹30,000       | Physically blocks blade-flash from radar line-of-sight                    |
| B1  | Radar-absorbing material wrap                | ₹2,000        | Reduces bulk RCS 10–20 dB                                                 |
| B3  | Corner-reflector decoy                       | ₹50 each      | Masks true micro-Doppler with a large false echo                          |
| B4  | Chaff dispenser                              | ₹500 + ₹100/use | Generates hundreds of false CFAR detections                              |
| C1  | Broadband noise jammer                       | ₹3,000        | Drops effective SNR below reliable detection range                        |
| C2  | DRFM ghost-target generator                  | ₹15,000       | Creates fake drones at false ranges                                       |
| C4  | Deceptive micro-Doppler injection            | ₹25,000       | Broadcasts synthetic bird micro-Doppler from a drone                      |
| D1  | Bird-mimicking flight pattern                | ₹0            | LSTM temporal features drift toward bird trajectory                       |
| D2  | Pulse-and-glide flight                       | ₹0            | Most 10-frame windows capture zero micro-Doppler                          |
| E1  | Flapping-wing ornithopter                    | ₹15,000       | Is physically indistinguishable from a bird to micro-Doppler              |

Attacks above the A/B/C/D/E divisions target different layers of the pipeline:

- **A-series:** Attack the Blade Flash Periodicity (BFP) feature
- **B-series:** Attack the CFAR detection stage (bulk RCS)
- **C-series:** Active electronic countermeasures
- **D-series:** Attack the LSTM temporal classifier
- **E-series:** Attack the premise of radar-only classification (requires multi-sensor fusion to counter)

---

## Implemented attacks

### A2: Fewer-blade drones (`attack_a2_fewer_blades.py`)

Tests whether drones with 1-blade propellers (instead of the 2-blade training distribution) evade the classifier. Runs the baseline CNN+LSTM+BFP model against progressively more aggressive variants:

| Variant              | n_blades | RPM  | Expected BFP | Hypothesis                        |
|----------------------|----------|------|--------------|-----------------------------------|
| clean_drone_control  | 2        | 5000 | 167 Hz       | Baseline sanity check             |
| A2_pure_1blade       | 1        | 5000 | 83 Hz        | Pure A2, normal RPM               |
| A2+A1_mild           | 1        | 3000 | 50 Hz        | Mild RPM reduction                |
| A2+A1_aggressive     | 1        | 2000 | 33 Hz        | BFP at edge of bird range         |
| A2+A1_extreme        | 1        | 1200 | 20 Hz        | BFP in bird range                 |
| A2+A1_bird_mimic     | 1        | 800  | 13 Hz        | BFP matches typical bird flap     |

Run: `python3 attack_a2_fewer_blades.py`

Output: `../baseline/results/attack_a2_results.json` with per-variant accuracy and confusion distribution.

---

## Planned next steps

1. Complete A2 evaluation
2. Implement A1 (pure RPM reduction), A3 (dielectric-coated blades)
3. Implement D2 (pulse-and-glide flight pattern)
4. Implement E1 (ornithopter micro-Doppler substitution) — the "existence proof" that radar-only classification is fundamentally insufficient
5. Evaluate adversarial training as a defence
6. Publish threat-cost taxonomy as a standalone paper
