from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import numpy as np
import os
import pandas as pd


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=True
):
    """
    Generate seq2seq samples with extended features for both x and y.
    """
    num_nodes = df["station"].nunique()
    base_values = df[["flow"] + [f"emb_{i}" for i in range(32)]].values
    data = np.reshape(base_values, newshape=[-1, num_nodes, 33])  # [N, nodes, 33]

    num_samples = data.shape[0]

    # ---------- Add time_in_day feature ----------
    if add_time_in_day:
        df['datatime'] = (
            pd.to_datetime(df.date)
            + pd.to_timedelta(df.hour, unit='h')
            + pd.to_timedelta(df.minute, unit='m')
        )
        time_ind = (df['datatime'].values - df['datatime'].values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = time_ind.reshape(-1, num_nodes, 1)
    else:
        print("1")
        time_in_day = None

    # ---------- Add day_of_week feature ----------
    if add_day_in_week:
        df['datatime'] = pd.to_datetime(df.date) + pd.to_timedelta(df.hour, unit='h') + pd.to_timedelta(df.minute,
                                                                                                        unit='m')
        # day_in_week as a proportion of the week: 0 for Monday, 6/6 for Sunday
        day_in_week = df['datatime'].dt.dayofweek.values / 6.0  # normalize to [0,1]
        day_in_week = day_in_week.reshape(-1, num_nodes, 1)  # [num_samples, num_nodes, 1]
    else:
        print("1")
        day_in_week = None

    # ---------- Generate seq2seq samples ----------
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = num_samples - abs(max(y_offsets))  # exclusive

    for t in range(min_t, max_t):
        # --- x features ---
        data_list_x = [data[t + x_offsets, ...]]
        if add_time_in_day:
            data_list_x.append(time_in_day[t + x_offsets, ...])
        if add_day_in_week:
            data_list_x.append(day_in_week[t + x_offsets, ...])
        x_t = np.concatenate(data_list_x, axis=-1)

        # --- y features ---
        data_list_y = [data[t + y_offsets, ...]]
        if add_time_in_day:
            data_list_y.append(time_in_day[t + y_offsets, ...])
        if add_day_in_week:
            data_list_y.append(day_in_week[t + y_offsets, ...])
        y_t = np.concatenate(data_list_y, axis=-1)

        x.append(x_t)
        y.append(y_t)

    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    print("Final x shape:", x.shape, "Final y shape:", y.shape)
    return x, y


def generate_train_val_test(args):
    df = pd.read_csv(args.traffic_df_filename)

    # Input 12 steps, predict next 12 steps
    x_offsets = np.sort(np.arange(-11, 1, 1))
    y_offsets = np.sort(np.arange(1, 13, 1))

    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=True,
    )

    print("x shape:", x.shape, ", y shape:", y.shape)

    # ---------- Split train / val / test ----------
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val]
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(f"{cat} x: {_x.shape}, y: {_y.shape}")
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


def main(args):
    print("Generating training data with extended features...")
    generate_train_val_test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="", help="Output directory.")
    parser.add_argument(
        "--traffic_df_filename",
        type=str,
        default=r"D:\python\pycharm\python项目\查看\traffic_with_embedding.csv",
        help="Raw traffic readings CSV file.",
    )
    args = parser.parse_args()
    main(args)
