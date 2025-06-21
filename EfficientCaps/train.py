import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from Dataset import get_mnist_dataloader
from Capsule import EfficientCaps
from Config import (
    Mnist_Train_Loader_Cfg,
    Mnist_Test_Loader_Cfg,
    Mnist_Training_Cfg,
)
from torchsummary import summary


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def Mnist_Train():

    # prepare the dataloder
    train_data_Cfg = Mnist_Train_Loader_Cfg()
    train_loader = get_mnist_dataloader(train_data_Cfg)
    test_data_Cfg = Mnist_Test_Loader_Cfg()
    test_loader = get_mnist_dataloader(test_data_Cfg)

    # load the model config
    model = EfficientCaps().to(DEVICE)

    # setting other hyperparameters
    train_Cfg = Mnist_Training_Cfg()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_Cfg.lr, betas=(train_Cfg.beta1, train_Cfg.beta2)
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=train_Cfg.step_size, gamma=train_Cfg.gamma
    )

    # summary the model
    summary(model, input_size=(1, 28, 28))

    # check whether we have resume to reload the model
    if train_Cfg.resume:
        checkpoint = torch.load(train_Cfg.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        train_Cfg.start_epoch = checkpoint["epoch"] + 1

    print("start epoch: ", train_Cfg.start_epoch + 1)

    # train the model on Mnist
    global_step = train_Cfg.start_epoch * len(train_loader)
    best_test_accuracy = 0.00
    for epoch in range(
        train_Cfg.start_epoch, train_Cfg.start_epoch + train_Cfg.num_epochs
    ):
        progress_bar = tqdm(
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{train_Cfg.num_epochs}",
        )
        train(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            num_classes=10,
            progress_bar=progress_bar,
            global_step=global_step,
        )
        progress_bar.close()
        lr_scheduler.step()

        # save the model
        if (
            epoch + 1
        ) % train_Cfg.save_model_epochs == 0 or epoch + 1 == train_Cfg.num_epochs:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "parameters": train_Cfg,
                "epoch": epoch,
            }
            output_dir_path = Path(train_Cfg.output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            model_save_path = output_dir_path / f"Mnist_epoch{epoch}.pth"
            torch.save(
                checkpoint,
                model_save_path,
            )
        best_test_accuracy = test(
            model=model,
            test_loader=test_loader,
            epoch=epoch,
            num_epochs=train_Cfg.num_epochs,
            best_test_accuracy=best_test_accuracy,
        )


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    num_classes: int,
    progress_bar: tqdm,
    global_step: int,
):

    model.train()
    train_loss = 0
    total_correct_predictions = 0
    total_samples = 0

    for step, (imgs, labels) in enumerate(train_loader):
        # convert the labels into one-hot format
        labels = torch.sparse.torch.eye(num_classes).index_select(dim=0, index=labels)
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        # forward and backward
        optimizer.zero_grad()
        out = model(imgs)
        loss = model.MarginalLoss(out, labels)
        loss.backward()
        optimizer.step()
        train_loss += (loss.detach().item() - train_loss) / (step + 1)

        # calculate the training accuracy
        class_lengths = out.norm(dim=-1)
        predicted_classes = torch.argmax(class_lengths, dim=-1)
        true_classes = torch.argmax(labels, dim=-1)
        total_correct_predictions += (predicted_classes == true_classes).sum().item()
        total_samples += labels.shape[0]
        train_accuracy = total_correct_predictions / total_samples

        progress_bar.update(1)
        logs = {
            "loss": train_loss,
            "train accuracy": train_accuracy,
            "lr": lr_scheduler.get_last_lr()[0],
            "step": global_step,
        }
        progress_bar.set_postfix(**logs)
        global_step += 1


def test(
    model: nn.Module,
    test_loader: torch.utils.data.dataloader,
    epoch: int,
    num_epochs: int,
    best_test_accuracy: float,
):
    model.eval()
    test_loss = 0
    total_correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch_id, (imgs, labels) in enumerate(test_loader):

            labels = torch.sparse.torch.eye(10).index_select(dim=0, index=labels)
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            out = model(imgs)
            loss = model.MarginalLoss(out, labels)
            test_loss += loss.item()

            # calculate the test accuracy
            class_lengths = out.norm(dim=-1)
            predicted_classes = torch.argmax(class_lengths, dim=-1)
            true_classes = torch.argmax(labels, dim=-1)
            total_correct_predictions += (
                (predicted_classes == true_classes).sum().item()
            )
            total_samples += labels.shape[0]

    accuracy = total_correct_predictions / total_samples
    best_test_accuracy = max(best_test_accuracy, accuracy)

    tqdm.write(
        "Epoch: [{}/{}], test accuracy: {:.6f}, test loss: {:.6f}, best test accuracy: {:.6f}".format(
            epoch + 1,
            num_epochs,
            accuracy,
            test_loss / len(test_loader),
            best_test_accuracy,
        )
    )
    return best_test_accuracy


if __name__ == "__main__":
    Mnist_Train()
