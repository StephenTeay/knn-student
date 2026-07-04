# Calibration notes for synthetic outcome generation

The coefficients in `generate_synthetic_data.py` that turn latent EI/engagement
traits into a final score were chosen empirically, not arbitrarily: a small
grid of (intercept, EI weight, engagement weight, interaction weight, noise
sd) combinations was tried against the same seeded latent-trait draws, and
the resulting at_risk / satisfactory / high split was checked against the
proposal's own stated expectation in Section 3.4.3 ("students at risk tend
to be a smaller class, maybe 20-30% of all enrolments").

Grid tried (high / satisfactory / at_risk):

| intercept | c_ei | c_eng | c_int | noise_sd | high | satisfactory | at_risk |
|-----------|------|-------|-------|----------|------|---------------|---------|
| 10        | 6    | 6     | 2     | 9        | 0.07 | 0.37          | 0.56    |
| 15        | 8    | 8     | 2     | 8        | 0.41 | 0.37          | 0.22    |  <- selected
| 20        | 8    | 8     | 2     | 8        | 0.48 | 0.40          | 0.12    |
| 18        | 9    | 9     | 3     | 8        | 0.56 | 0.35          | 0.09    |

The selected row is the one whose at_risk share falls inside the 20-30%
band the proposal cites, while still leaving "high" as a plausible plurality
class rather than an extreme majority. This is a modeling convenience for
the synthetic demo, not a finding -- a real fielded dataset will have
whatever class balance it has, and the ETL/feature/model code does not
assume this specific split (SMOTE and stratified CV are both designed to
work across a range of imbalance ratios).
