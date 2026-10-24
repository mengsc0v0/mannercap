"""Step 6 intake acceptance set.  Written before any model output on it was seen; frozen after one run.
Mix: clear requests that must be accepted (incl. clear pronouns and single-plausible-candidate pronouns), true ambiguities,
unsupported requests and task-program conflicts.  Gold is hand-written against the registered programs only."""


def cases(case, F, program_steps):
    S = lambda t: [list(s) for s in program_steps(F.program(t))]  # noqa: E731
    G = lambda role, lvl: (f"{role}.pick_1.grasp", "grip", lvl, role)  # noqa: E731
    out = [
        # ---- accept: explicit object, task restated
        case("A01", 1, "Put the cream cheese box and the butter in the basket. Hold the butter firmly.", "accept", bind=[G("butter", "firm")], steps=S(1)),
        case("A02", 0, "put the alphabet soup and the tomato sauce in the basket, grasp the tomato sauce gently", "accept", bind=[G("tomato_sauce", "gentle")], steps=S(0)),
        case("A04", 4, "put the white mug on the left plate and the yellow and white mug on the right plate, holding the yellow and white mug firmly", "accept", bind=[G("yellow_white_mug", "firm")], steps=S(4)),
        case("A05", 6, "Put the white mug on the plate. Then put the chocolate pudding to the right of the plate. Hold the mug gently.", "accept", bind=[G("white_mug", "gentle")], steps=S(6)),
        case("A06", 8, "put both moka pots on the stove; hold the left moka pot gently", "accept", bind=[G("left_moka_pot", "gentle")], steps=S(8)),
        case("A07", 9, "put the yellow and white mug in the microwave and close the microwave slowly", "accept", bind=[("microwave.close_1.articulate", "speed", "slow", "microwave")], steps=S(9)),
        case("A08", 3, "Put the black bowl in the bottom drawer of the cabinet. Close the drawer fast.", "accept", bind=[("drawer.close_1.articulate", "speed", "fast", "drawer")], steps=S(3)),
        case("A09", 2, "Turn on the stove slowly and put the moka pot on the stove.", "accept", bind=[("stove.turn_on_1.articulate", "speed", "slow", "stove")], steps=S(2)),
        case("A10", 5, "Pick up the book and place it in the back compartment of the caddy. Hold the book firmly.", "accept", bind=[G("book", "firm")], steps=S(5)),
        case("A16", 0, "Put the alphabet soup in the basket and the tomato sauce in the basket. Grip the alphabet soup gently and the tomato sauce firmly.", "accept",
             bind=[G("alphabet_soup", "gentle"), G("tomato_sauce", "firm")], steps=S(0)),
        case("A19", 0, "put the alphabet soup and the tomato sauce in the basket and hold both of them gently", "accept",
             bind=[G("alphabet_soup", "gentle"), G("tomato_sauce", "gentle")], steps=S(0)),
        # ---- accept: pronoun with a clear antecedent / single plausible candidate / no object named but one candidate
        case("A03", 7, "Pick up the cream cheese box. Hold it gently.", "accept", bind=[G("cream_cheese", "gentle")], steps=[], why="clear antecedent"),
        case("A15", 1, "Pick up the butter. Hold it gently.", "accept", bind=[G("butter", "gentle")], steps=[], why="clear antecedent"),
        case("A12", 9, "put the yellow and white mug in the microwave and close it, hold it gently", "accept", bind=[G("yellow_white_mug", "gentle")], steps=S(9),
             why="'hold it': only the mug can be held (the microwave has no grasp stage)"),
        case("A13", 3, "put the black bowl in the bottom drawer of the cabinet and close it; hold it firmly", "accept", bind=[G("black_bowl", "firm")], steps=S(3),
             why="only the bowl can be held"),
        case("A14", 2, "turn on the stove and put the moka pot on it. Hold it gently.", "accept", bind=[G("moka_pot", "gentle")], steps=S(2),
             why="only the moka pot can be held"),
        case("A11", 5, "Hold it firmly.", "accept", bind=[G("book", "firm")], steps=None, why="single manipulated object"),
        case("A18", 5, "抓取的时候轻一点", "accept", bind=[G("book", "gentle")], steps=None, why="manner only; registered task has one grasp"),
        # ---- ambiguous
        case("B01", 1, "put the cream cheese box and the butter in the basket; hold one of them gently", "ambiguous", roles=["cream_cheese", "butter"]),
        case("B02", 7, "Put the alphabet soup and the cream cheese box in the basket. Grip it firmly.", "ambiguous", roles=["alphabet_soup", "cream_cheese"]),
        case("B03", 8, "put both moka pots on the stove; hold the pot firmly", "ambiguous", roles=["right_moka_pot", "left_moka_pot"]),
        case("B04", 4, "put the white mug on the left plate and the yellow and white mug on the right plate; grip it gently", "ambiguous", roles=["white_mug", "yellow_white_mug"]),
        case("B05", 6, "put the white mug on the plate and the chocolate pudding to the right of the plate, then hold it gently", "ambiguous", roles=["white_mug", "chocolate_pudding"]),
        case("B06", 7, "put both the alphabet soup and the cream cheese box in the basket; go slowly when approaching", "ambiguous", roles=["alphabet_soup", "cream_cheese"]),
        case("B07", 9, "put the yellow and white mug in the microwave and close it; go slowly", "ambiguous", roles=["yellow_white_mug", "microwave"]),
        # ---- unsupported
        case("U01", 5, "Pick up the book and place it in the back compartment of the caddy. Hold the book with 20 N of force.", "unsupported", code="numeric_level"),
        case("U02", 1, "put the cream cheese box and the butter in the basket; hold the butter at 40% grip", "unsupported", code="numeric_level"),
        case("U03", 3, "Put the black bowl in the bottom drawer of the cabinet and close it, squeezing the drawer gently", "unsupported", code="dimension_not_on_stage"),
        case("U04", 9, "put the yellow and white mug in the microwave and close it without any impact", "unsupported", code="quantity_not_measured"),
        case("U05", 6, "put the white mug on the plate and the chocolate pudding to the right of the plate; hold the pudding gently and move the mug at 3 cm/s", "unsupported",
             code="numeric_level", why="partial: the gentle part is executable but the request as a whole is not"),
        case("U06", 0, "put the alphabet soup and the tomato sauce in the basket; grip the cream cheese firmly", "unsupported", code="object_not_in_task"),
        # ---- conflict (task text vs registered program)
        case("C01", 7, "put the cream cheese box in the basket and then the alphabet soup in the basket; hold the soup gently", "conflict", code="order_mismatch"),
        case("C02", 3, "put the black bowl on top of the cabinet and close the drawer", "conflict", code="target_not_in_program"),
        case("C03", 6, "put the white mug on the plate and the chocolate pudding on the plate; hold the pudding firmly", "conflict", code="target_mismatch"),
        case("C04", 4, "put the red mug on the left plate and the yellow and white mug on the right plate", "conflict", code="object_not_in_program"),
        case("C05", 0, "put the alphabet soup and the ketchup in the basket; hold the soup gently", "conflict", code="object_not_in_program"),
        case("C06", 2, "put the moka pot on the stove and then turn on the stove", "conflict", code="order_mismatch"),
        case("C07", 9, "put the yellow and white mug in the microwave and open it", "conflict", code="step_not_in_program"),
    ]
    return out
