from pandas import read_csv

if __name__ == "__main__":
    df = read_csv("examples/02_example_transform.csv", index_col=0)

    is_ternary = True

    # Отбираем негативные отзывы
    df = df[df["label"] == 1 + is_ternary]
    df.to_csv("examples/02a_example_filtered_data.csv")
