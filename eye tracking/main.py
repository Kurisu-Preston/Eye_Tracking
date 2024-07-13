from data_loader import load_data
from model import train_model, build_model
# from evaluate import evaluate_model
# from plot import plot_loss


def main():
    # CSV file path and test set size
    csv_path = '/Users/chris/Desktop/data/data_base.csv'
    # test_size = 0.1

    # Load and split data
    X_train, y_train = load_data(csv_path)

    # # Build and train the model
    # model, log = train_model(X_train, y_train)

    # 构建模型
    input_dim = 4
    model = build_model(input_dim)

    # 设置训练参数
    batch_size = 32
    epochs = 100

    # 启动训练
    train_model(model, X_train, y_train, batch_size, epochs)

    # # Evaluate model
    # evaluate_model(model, x_test, y_test)
    #
    # # Plot loss graph
    # plot_loss(log)


if __name__ == "__main__":
    main()
