with open("example/example.txt", "r") as f:
    f_lines = f.readlines()
    with open("terms_processed.txt", "r") as g:
        g_lines = g.readlines()
        # print(g_lines)
        for g_term in g_lines:
            g_term = g_term.split(":")[0]
            g_term = g_term.strip().lower()

            for f_term in f_lines:
                f_term = f_term.strip().lower()
                if g_term == f_term:
                    print("Repeat: ", f_term, g_term)
                    break
