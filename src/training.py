from torch.optim.lr_scheduler import CosineAnnealingLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

scheduler = CosineAnnealingLR(optimizer, T_max=55, eta_min=1e-6)
# The learning deceleration cycle lasts exactly the same amount as the total number of Epochs (40 cycles).
# eta_min=1e-6: The minimum learning rate that the system is allowed to reduce to.

EPOCHS = 55
best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train() # turn on train mode
    total_loss, correct, n = 0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad() # delete memories
        logits = model(imgs) # predict
        loss = criterion(logits, labels) # validate
        loss.backward() # calculate gradient (learn from experience)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step() # update weight
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        n += len(labels)
    scheduler.step()

    # Validation phase
    # After completing an Epoch, the model must take a test to see if it has learned by rote.
    model.eval()
    val_correct, val_n = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            val_correct += (preds == labels).sum().item()
            val_n += len(labels)

    # Save the best model
    val_acc = val_correct / val_n
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
    print(f"Ep {epoch+1:3d}  loss={total_loss/n:.4f}  "
          f"train={correct/n:.3f}  val={val_acc:.3f}")
