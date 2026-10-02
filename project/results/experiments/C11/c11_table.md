| case | paretoset 1.2.0 | paretoset 1.2.5 | same |
|---|---|---|---|
| `S_expected_front(brute force)` | ["P1", "P2", "P4", "P5"] | ["P1", "P2", "P4", "P5"] | yes |
| `S_lower_DataFrame` | ["P1", "P2", "P4", "P5"] | ["P1", "P2", "P4", "P5"] | yes |
| `S_lower_ndarray` | ["P1", "P2", "P4", "P5"] | ["P1", "P2", "P4", "P5"] | yes |
| `S_Capitalised_DataFrame` | ["P1", "P2", "P4", "P5"] | ["P1", "P2", "P4", "P5"] | yes |
| `S_Capitalised_ndarray` | ["P1", "P2", "P4", "P5"] | ["P1", "P2", "P4", "P5"] | yes |
| `S_UPPER_DataFrame` | ["P1", "P2", "P4", "P5"] | ["P1", "P2", "P4", "P5"] | yes |
| `S_UPPER_ndarray` | ["P1", "P2", "P4", "P5"] | ["P1", "P2", "P4", "P5"] | yes |
| `S_sense_None(all min)` | ["P1", "P3", "P4", "P5"] | ["P1", "P3", "P4", "P5"] | yes |
| `S_input_left_unchanged` | true | true | yes |
| `D1_guide_example|distinct=True|numba=True` | ["a", "c", "d"] | ["a", "c", "d"] | yes |
| `D1_guide_example|distinct=True|numba=False` | ["a", "c", "d"] | ["a", "c", "d"] | yes |
| `D1_guide_example|paretorank|distinct=True` | {"a": 1, "b": 2, "c": 1, "d": 1} | {"a": 1, "b": 2, "c": 1, "d": 1} | yes |
| `D1_guide_example|distinct=False|numba=True` | ["a", "b", "c", "d"] | ["a", "b", "c", "d"] | yes |
| `D1_guide_example|distinct=False|numba=False` | ["a", "b", "c", "d"] | ["a", "b", "c", "d"] | yes |
| `D1_guide_example|paretorank|distinct=False` | {"a": 1, "b": 1, "c": 1, "d": 1} | {"a": 1, "b": 1, "c": 1, "d": 1} | yes |
| `D2_three_objectives|distinct=True|numba=True` | ["A", "C"] | ["A", "C"] | yes |
| `D2_three_objectives|distinct=True|numba=False` | ["A", "C"] | ["A", "C"] | yes |
| `D2_three_objectives|paretorank|distinct=True` | {"A": 1, "B": 2, "C": 1, "D": 4, "E": 5, "F": 3} | {"A": 1, "B": 2, "C": 1, "D": 4, "E": 5, "F": 3} | yes |
| `D2_three_objectives|distinct=False|numba=True` | ["A", "B", "C", "F"] | ["A", "B", "C", "F"] | yes |
| `D2_three_objectives|distinct=False|numba=False` | ["A", "B", "C", "F"] | ["A", "B", "C", "F"] | yes |
| `D2_three_objectives|paretorank|distinct=False` | {"A": 1, "B": 1, "C": 1, "D": 2, "E": 2, "F": 1} | {"A": 1, "B": 1, "C": 1, "D": 2, "E": 2, "F": 1} | yes |
| `D3_all_identical|distinct=True|numba=True` | ["a"] | ["a"] | yes |
| `D3_all_identical|distinct=True|numba=False` | ["a"] | ["a"] | yes |
| `D3_all_identical|paretorank|distinct=True` | {"a": 1, "b": 2, "c": 3, "d": 4} | {"a": 1, "b": 2, "c": 3, "d": 4} | yes |
| `D3_all_identical|distinct=False|numba=True` | ["a", "b", "c", "d"] | ["a", "b", "c", "d"] | yes |
| `D3_all_identical|distinct=False|numba=False` | ["a", "b", "c", "d"] | ["a", "b", "c", "d"] | yes |
| `D3_all_identical|paretorank|distinct=False` | {"a": 1, "b": 1, "c": 1, "d": 1} | {"a": 1, "b": 1, "c": 1, "d": 1} | yes |
| `D4_tie_on_two_of_three|distinct=True|numba=True` | ["a"] | ["a"] | yes |
| `D4_tie_on_two_of_three|distinct=True|numba=False` | ["a"] | ["a"] | yes |
| `D4_tie_on_two_of_three|paretorank|distinct=True` | {"a": 1, "b": 2} | {"a": 1, "b": 2} | yes |
| `D4_tie_on_two_of_three|distinct=False|numba=True` | ["a"] | ["a"] | yes |
| `D4_tie_on_two_of_three|distinct=False|numba=False` | ["a"] | ["a"] | yes |
| `D4_tie_on_two_of_three|paretorank|distinct=False` | {"a": 1, "b": 2} | {"a": 1, "b": 2} | yes |
| `N1_nan_first_objective|distinct=True|numba=True` | ["b"] | ["b"] | yes |
| `N1_nan_first_objective|distinct=True|numba=True|all_row_orders` | ["b"] | ["b"] | yes |
| `N1_nan_first_objective|distinct=True|numba=False` | ["b"] | ["b"] | yes |
| `N1_nan_first_objective|distinct=True|numba=False|all_row_orders` | ["b"] | ["b"] | yes |
| `N1_nan_first_objective|distinct=False|numba=True` | ["b"] | ["b"] | yes |
| `N1_nan_first_objective|distinct=False|numba=True|all_row_orders` | ["b"] | ["b"] | yes |
| `N1_nan_first_objective|distinct=False|numba=False` | ["b"] | ["b"] | yes |
| `N1_nan_first_objective|distinct=False|numba=False|all_row_orders` | ["b"] | ["b"] | yes |
| `N1_nan_first_objective|remedy=drop_nan_rows` | ["a"] | ["a"] | yes |
| `N1_nan_first_objective|remedy=fill_inf` | ["a", "b"] | ["a", "b"] | yes |
| `N1_nan_first_objective|remedy=fill_inf|equals_brute_force` | true | true | yes |
| `N2_nan_second_objective|distinct=True|numba=True` | ["b"] | ["b"] | yes |
| `N2_nan_second_objective|distinct=True|numba=True|all_row_orders` | ["b"] | ["b"] | yes |
| `N2_nan_second_objective|distinct=True|numba=False` | ["b"] | ["b"] | yes |
| `N2_nan_second_objective|distinct=True|numba=False|all_row_orders` | ["b"] | ["b"] | yes |
| `N2_nan_second_objective|distinct=False|numba=True` | ["b"] | ["b"] | yes |
| `N2_nan_second_objective|distinct=False|numba=True|all_row_orders` | ["b"] | ["b"] | yes |
| `N2_nan_second_objective|distinct=False|numba=False` | ["b"] | ["b"] | yes |
| `N2_nan_second_objective|distinct=False|numba=False|all_row_orders` | ["b"] | ["b"] | yes |
| `N2_nan_second_objective|remedy=drop_nan_rows` | ["a"] | ["a"] | yes |
| `N2_nan_second_objective|remedy=fill_inf` | ["a", "b"] | ["a", "b"] | yes |
| `N2_nan_second_objective|remedy=fill_inf|equals_brute_force` | true | true | yes |
| `N3_all_nan_row|distinct=True|numba=True` | ["a"] | ["a"] | yes |
| `N3_all_nan_row|distinct=True|numba=True|all_row_orders` | ["a", "b"] | ["a", "b"] | yes |
| `N3_all_nan_row|distinct=True|numba=False` | ["a"] | ["a"] | yes |
| `N3_all_nan_row|distinct=True|numba=False|all_row_orders` | ["a", "b"] | ["a", "b"] | yes |
| `N3_all_nan_row|distinct=False|numba=True` | ["a"] | ["a"] | yes |
| `N3_all_nan_row|distinct=False|numba=True|all_row_orders` | ["a", "b"] | ["a", "b"] | yes |
| `N3_all_nan_row|distinct=False|numba=False` | ["a"] | ["a"] | yes |
| `N3_all_nan_row|distinct=False|numba=False|all_row_orders` | ["a", "b"] | ["a", "b"] | yes |
| `N3_all_nan_row|remedy=drop_nan_rows` | ["b"] | ["b"] | yes |
| `N3_all_nan_row|remedy=fill_inf` | ["b"] | ["b"] | yes |
| `N3_all_nan_row|remedy=fill_inf|equals_brute_force` | true | true | yes |
| `N4_pattern_in_all_cases|distinct=True|numba=True` | ["p1", "p2", "p4", "p5"] | ["p1", "p2", "p4", "p5"] | yes |
| `N4_pattern_in_all_cases|distinct=True|numba=True|all_row_orders` | ["p1,p2,p4,p5"] | ["p1,p2,p4,p5"] | yes |
| `N4_pattern_in_all_cases|distinct=True|numba=False` | ["p1", "p2", "p4", "p5"] | ["p1", "p2", "p4", "p5"] | yes |
| `N4_pattern_in_all_cases|distinct=True|numba=False|all_row_orders` | ["p1,p2,p4,p5"] | ["p1,p2,p4,p5"] | yes |
| `N4_pattern_in_all_cases|distinct=False|numba=True` | ["p1", "p2", "p4", "p5"] | ["p1", "p2", "p4", "p5"] | yes |
| `N4_pattern_in_all_cases|distinct=False|numba=True|all_row_orders` | ["p1,p2,p4,p5"] | ["p1,p2,p4,p5"] | yes |
| `N4_pattern_in_all_cases|distinct=False|numba=False` | ["p1", "p2", "p4", "p5"] | ["p1", "p2", "p4", "p5"] | yes |
| `N4_pattern_in_all_cases|distinct=False|numba=False|all_row_orders` | ["p1,p2,p4,p5"] | ["p1,p2,p4,p5"] | yes |
| `N4_pattern_in_all_cases|remedy=drop_nan_rows` | ["p1", "p2", "p5"] | ["p1", "p2", "p5"] | yes |
| `N4_pattern_in_all_cases|remedy=fill_1.0` | ["p1", "p2", "p4", "p5"] | ["p1", "p2", "p4", "p5"] | yes |
| `N4_pattern_in_all_cases|remedy=fill_1.0|equals_brute_force` | true | true | yes |
| `N4_pattern_in_all_cases|remedy=fill_inf` | ["p1", "p2", "p4", "p5"] | ["p1", "p2", "p4", "p5"] | yes |
| `N4_pattern_in_all_cases|remedy=fill_inf|equals_brute_force` | true | true | yes |
| `N5_nan_row_worse_elsewhere|distinct=True|numba=True` | ["q1"] | ["q1"] | yes |
| `N5_nan_row_worse_elsewhere|distinct=True|numba=True|all_row_orders` | ["q1"] | ["q1"] | yes |
| `N5_nan_row_worse_elsewhere|distinct=True|numba=False` | ["q1"] | ["q1"] | yes |
| `N5_nan_row_worse_elsewhere|distinct=True|numba=False|all_row_orders` | ["q1"] | ["q1"] | yes |
| `N5_nan_row_worse_elsewhere|distinct=False|numba=True` | ["q1"] | ["q1"] | yes |
| `N5_nan_row_worse_elsewhere|distinct=False|numba=True|all_row_orders` | ["q1"] | ["q1"] | yes |
| `N5_nan_row_worse_elsewhere|distinct=False|numba=False` | ["q1"] | ["q1"] | yes |
| `N5_nan_row_worse_elsewhere|distinct=False|numba=False|all_row_orders` | ["q1"] | ["q1"] | yes |
| `N5_nan_row_worse_elsewhere|remedy=drop_nan_rows` | ["q1"] | ["q1"] | yes |
| `N5_nan_row_worse_elsewhere|remedy=fill_1.0` | ["q1"] | ["q1"] | yes |
| `N5_nan_row_worse_elsewhere|remedy=fill_1.0|equals_brute_force` | true | true | yes |
| `N5_nan_row_worse_elsewhere|remedy=fill_inf` | ["q1"] | ["q1"] | yes |
| `N5_nan_row_worse_elsewhere|remedy=fill_inf|equals_brute_force` | true | true | yes |
| `N6_nan_row_hides_better_distance|distinct=True|numba=True` | ["r_all"] | ["r_all"] | yes |
| `N6_nan_row_hides_better_distance|distinct=True|numba=True|all_row_orders` | ["r_all"] | ["r_all"] | yes |
| `N6_nan_row_hides_better_distance|distinct=True|numba=False` | ["r_all"] | ["r_all"] | yes |
| `N6_nan_row_hides_better_distance|distinct=True|numba=False|all_row_orders` | ["r_all"] | ["r_all"] | yes |
| `N6_nan_row_hides_better_distance|distinct=False|numba=True` | ["r_all"] | ["r_all"] | yes |
| `N6_nan_row_hides_better_distance|distinct=False|numba=True|all_row_orders` | ["r_all"] | ["r_all"] | yes |
| `N6_nan_row_hides_better_distance|distinct=False|numba=False` | ["r_all"] | ["r_all"] | yes |
| `N6_nan_row_hides_better_distance|distinct=False|numba=False|all_row_orders` | ["r_all"] | ["r_all"] | yes |
| `N6_nan_row_hides_better_distance|remedy=drop_nan_rows` | ["r_x", "r_y"] | ["r_x", "r_y"] | yes |
| `N6_nan_row_hides_better_distance|remedy=fill_1.0` | ["r_all", "r_x", "r_y"] | ["r_all", "r_x", "r_y"] | yes |
| `N6_nan_row_hides_better_distance|remedy=fill_1.0|equals_brute_force` | true | true | yes |
| `N6_nan_row_hides_better_distance|remedy=fill_inf` | ["r_all", "r_x", "r_y"] | ["r_all", "r_x", "r_y"] | yes |
| `N6_nan_row_hides_better_distance|remedy=fill_inf|equals_brute_force` | true | true | yes |
| `X_random_500x3_round1|duplicate_rows` | 103 | 103 | yes |
| `X_random_500x3_round1|brute_force_front_size` | 6 | 6 | yes |
| `X_random_500x3_round1|numba=True|distinct=False|front_size` | 6 | 6 | yes |
| `X_random_500x3_round1|numba=True|distinct=False|equals_brute_force` | true | true | yes |
| `X_random_500x3_round1|numba=True|distinct=True|front_size` | 6 | 6 | yes |
| `X_random_500x3_round1|numba=True|distinct=True|equals_first_occurrence_of_brute_force` | true | true | yes |
| `X_random_500x3_round1|numba=True|mask_sha1` | "6c30a078cbefe6cbaed28b51aff0b74aa1919400" | "6c30a078cbefe6cbaed28b51aff0b74aa1919400" | yes |
| `X_random_500x3_round1|numba=False|distinct=False|front_size` | 6 | 6 | yes |
| `X_random_500x3_round1|numba=False|distinct=False|equals_brute_force` | true | true | yes |
| `X_random_500x3_round1|numba=False|distinct=True|front_size` | 6 | 6 | yes |
| `X_random_500x3_round1|numba=False|distinct=True|equals_first_occurrence_of_brute_force` | true | true | yes |
| `X_random_500x3_round1|numba=False|mask_sha1` | "6c30a078cbefe6cbaed28b51aff0b74aa1919400" | "6c30a078cbefe6cbaed28b51aff0b74aa1919400" | yes |
| `X_random_500x3_round1|paretorank|distinct=True|n_layers` | 36 | 36 | yes |
| `X_random_500x3_round1|paretorank|distinct=True|layer1_equals_paretoset` | true | true | yes |
| `X_random_500x3_round1|paretorank|distinct=True|ranks_sha1` | "5fec22e66a2c00981490f80c8ed8a5ac52f52096" | "5fec22e66a2c00981490f80c8ed8a5ac52f52096" | yes |
| `X_random_500x3_round1|paretorank|distinct=False|n_layers` | 25 | 25 | yes |
| `X_random_500x3_round1|paretorank|distinct=False|layer1_equals_paretoset` | true | true | yes |
| `X_random_500x3_round1|paretorank|distinct=False|ranks_sha1` | "adb792e5441e236d4dc26f7b4b8a62022ef2b4b5" | "adb792e5441e236d4dc26f7b4b8a62022ef2b4b5" | yes |
| `X_random_2000x3_round2|duplicate_rows` | 2 | 2 | yes |
| `X_random_2000x3_round2|brute_force_front_size` | 19 | 19 | yes |
| `X_random_2000x3_round2|numba=True|distinct=False|front_size` | 19 | 19 | yes |
| `X_random_2000x3_round2|numba=True|distinct=False|equals_brute_force` | true | true | yes |
| `X_random_2000x3_round2|numba=True|distinct=True|front_size` | 19 | 19 | yes |
| `X_random_2000x3_round2|numba=True|distinct=True|equals_first_occurrence_of_brute_force` | true | true | yes |
| `X_random_2000x3_round2|numba=True|mask_sha1` | "c1e5f9598e2abfe4d941277c3e607123cfd8be5c" | "c1e5f9598e2abfe4d941277c3e607123cfd8be5c" | yes |
| `X_random_2000x3_round2|numba=False|distinct=False|front_size` | 19 | 19 | yes |
| `X_random_2000x3_round2|numba=False|distinct=False|equals_brute_force` | true | true | yes |
| `X_random_2000x3_round2|numba=False|distinct=True|front_size` | 19 | 19 | yes |
| `X_random_2000x3_round2|numba=False|distinct=True|equals_first_occurrence_of_brute_force` | true | true | yes |
| `X_random_2000x3_round2|numba=False|mask_sha1` | "c1e5f9598e2abfe4d941277c3e607123cfd8be5c" | "c1e5f9598e2abfe4d941277c3e607123cfd8be5c" | yes |
| `X_random_2000x3_round2|paretorank|distinct=True|n_layers` | 29 | 29 | yes |
| `X_random_2000x3_round2|paretorank|distinct=True|layer1_equals_paretoset` | true | true | yes |
| `X_random_2000x3_round2|paretorank|distinct=True|ranks_sha1` | "4c76a4e77297beeab1eca5b8fe017e52a1bfbafc" | "4c76a4e77297beeab1eca5b8fe017e52a1bfbafc" | yes |
| `X_random_2000x3_round2|paretorank|distinct=False|n_layers` | 29 | 29 | yes |
| `X_random_2000x3_round2|paretorank|distinct=False|layer1_equals_paretoset` | true | true | yes |
| `X_random_2000x3_round2|paretorank|distinct=False|ranks_sha1` | "3b0478714780813a6822d955177dcda7fda70c4e" | "3b0478714780813a6822d955177dcda7fda70c4e" | yes |
| `V_sense_diff_with_DataFrame` | "ERROR KeyError: ('g1',)" | [true, false, true, false] | **NO** |
| `V_crowding_distance` | "ERROR AttributeError: module 'numpy' has no attribute 'float'.\n`np.float` was a deprecated alias for the builtin `float`. To avoid this error " | [Infinity, 0.4, 0.6, 0.6, Infinity] | **NO** |
