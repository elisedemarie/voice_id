from src import pipeline
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice ID Classification Pipeline")
    parser.add_argument("command", choices=["prepare", "train", "eval"], help="Command to execute")
    parser.add_argument("-xgb", action="store_true", help="Use XGBoost model")
    parser.add_argument("-cnn", action="store_true", help="Use CNN model")
    parser.add_argument("-val", action="store_true", help="Evaluate on validation set")
    parser.add_argument("-test", action="store_true", help="Evaluate on test set")
    args = parser.parse_args()

    if args.command == "prepare":
        pipeline.prepare()
    elif args.command == "train":
        if args.xgb:
            pipeline.train("xgb")
        elif args.cnn:
            pipeline.train("cnn")
        else:
            raise ValueError("Please specify a model type. Use -xgb or -cnn.")
    elif args.command == "eval":
        if args.val:
            if args.xgb:
                pipeline.eval('val', 'xgb')
            elif args.cnn:
                pipeline.eval('val', 'cnn')
            else:
                raise ValueError("Please specify a model type. Use -xgb or -cnn.")
        elif args.test:
            if args.xgb:
                pipeline.eval('test', 'xgb')
            elif args.cnn:
                pipeline.eval( 'test', 'cnn')
            else:
                raise ValueError("Please specify a model type. Use -xgb or -cnn.")
        else:
            raise ValueError("Please specify a set type. Use -val or -test.")