import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import time
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import torch.nn.functional as F
from sklearn.metrics.pairwise import euclidean_distances

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train_model(model, train_loader,criterion,optimizer,scheduler,epochs):
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)
        start_time = time.time()
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        scheduler.step()
        end_time = time.time()
        print('TrainLoss: %.3f | TrainAcc: %.3f%% (%d/%d) | Time Elapsed %.3f sec' % (
            train_loss / (batch_idx + 1), 100. * correct / total, correct, total, end_time - start_time))

# Test loop
def test_model(model, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print('TestLoss: %.3f | TestAcc: %.3f%% (%d/%d)' % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    return 100. * correct / total

def calculate_cluster_centers(embeddings, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    cluster_centers = kmeans.cluster_centers_
    return cluster_centers

def get_most_diverse_samples(tsne_results, cluster_centers, num_diverse_samples):
    distances = euclidean_distances(tsne_results, cluster_centers)
    min_distances = np.max(distances, axis=1)
    sorted_indices = np.argsort(min_distances)
    diverse_indices = sorted_indices[:num_diverse_samples]
    return diverse_indices

def extract_embeddings(model, test):
    test_loader = data.DataLoader(test, batch_size=64, shuffle=False)
    model.eval()
    embeddings = []
    targets_list = []
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            intermediate_features = model(images)
            embeddings.extend(intermediate_features.view(intermediate_features.size(0), -1).tolist())
            targets_list.append(targets)
    return embeddings

def least_confidence_images(model, test_dataset, k=2500):
    test_loader = data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    confidences = []
    labels = []
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            confidences.extend(max_probs.cpu().tolist())
            labels.extend(targets.cpu().tolist())
    confidences = torch.tensor(confidences)
    _, indices = torch.topk(confidences, k, largest=False)
    return data.Subset(test_dataset, indices), [labels[i] for i in indices]


def high_confidence_images(model, test_dataset, k=2500):
    test_loader = data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    confidences = []
    labels = []
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            confidences.extend(max_probs.cpu().tolist())
            labels.extend(targets.cpu().tolist())
    confidences = torch.tensor(confidences)
    _, indices = torch.topk(confidences, k, largest=True)
    return data.Subset(test_dataset, indices), [labels[i] for i in indices]


def LC_HC_diverse(embed_model, remainder,n=None):
    least_conf_images, least_conf_labels = least_confidence_images(embed_model, remainder, int(n/2))
    high_conf_images, high_conf_labels = high_confidence_images(embed_model, remainder, k=len(remainder) if 2*n > len(remainder) else 2*n)
    embeddings = extract_embeddings(embed_model, high_conf_images)
    new_embeddings = np.array(embeddings)
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(new_embeddings)
    cluster_centers = calculate_cluster_centers(tsne_results, 10)
    diverse_indices = get_most_diverse_samples(tsne_results, cluster_centers, int(n/2))
    ds = [data.Subset(high_conf_images, diverse_indices), least_conf_images]
    return data.ConcatDataset(ds)


def HC_diverse(embed_model,remainder, n=None):
    high_conf_images, high_conf_labels = high_confidence_images(embed_model, remainder, k=len(remainder) if 2*n > len(remainder) else 2*n)
    embeddings = extract_embeddings(embed_model, high_conf_images)
    new_embeddings = np.array(embeddings)
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(new_embeddings)
    cluster_centers = calculate_cluster_centers(tsne_results, 10)
    diverse_indices = get_most_diverse_samples(tsne_results, cluster_centers, n)
    return data.Subset(high_conf_images, diverse_indices)


def LC_diverse(embed_model, remainder, n=None):
    least_conf_images, least_conf_labels = least_confidence_images(embed_model, remainder, k=len(remainder) if 2*n > len(remainder) else 2*n)
    embeddings = extract_embeddings(embed_model, least_conf_images)
    new_embeddings = np.array(embeddings)
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(new_embeddings)
    cluster_centers = calculate_cluster_centers(tsne_results, 10)
    diverse_indices = get_most_diverse_samples(tsne_results, cluster_centers, n)
    return data.Subset(least_conf_images, diverse_indices)


def train_until_empty(model, initial_train_set, ini, remainder_set, test_set, max_iterations=15, batch_size=64, learning_rate=0.01, method=1):
    exp_acc = [ini]

    for iteration in range(max_iterations):
        if method == 1:
            train_data = LC_HC_diverse(model, remainder_set, n=5000)
        elif method == 2:
            train_data = HC_diverse(model, remainder_set, n=5000)
        elif method == 3:
            train_data = LC_diverse(model, remainder_set, n=5000)
        else:
            print("Invalid Method")
            return exp_acc

        if len(remainder_set) == 0:
            print("Dataset is empty. Stopping training.")
            break

        selected_indices = [remainder_set[i] for i in train_data.indices]
        remainder_set = list(set(remainder_set) - set(selected_indices))

        initial_train_set = data.ConcatDataset([initial_train_set, train_data])

        print(f"\nTraining iteration {iteration + 1}")
        print(f"Train Set Size: {len(initial_train_set)}, Remainder Size: {len(remainder_set)}")
        train_loader = data.DataLoader(initial_train_set, batch_size=batch_size, shuffle=True)
        train_model(model, train_loader, epochs=50, learning_rate=learning_rate)

        test_loader = data.DataLoader(test_set, batch_size=batch_size)
        accuracy = test_model(model, test_loader)
        exp_acc.append(accuracy)

        print(f"Iteration {iteration + 1}: Test Accuracy - {accuracy:.2f}")

        return exp_acc

        print(f"\nTraining iteration {iteration + 1}")
        print(f"Train Set Size: {len(initial_train_set)}, Remainder Size: {len(remainder_set)}")
        train_loader = data.DataLoader(initial_train_set, batch_size=batch_size, shuffle=True)
        train_model(model, train_loader, epochs=50, learning_rate=learning_rate)

        test_loader = data.DataLoader(test_set, batch_size=batch_size)
        accuracy = test_model(model, test_loader)
        exp_acc.append(accuracy)

        print(f"Iteration {iteration + 1}: Test Accuracy - {accuracy:.2f}")

    return exp_acc