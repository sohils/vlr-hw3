from torch.utils.data import DataLoader
import torch

from tensorboardX import SummaryWriter


class ExperimentRunnerBase(object):
    """
    This base class contains the simple train and validation loops for your VQA experiments.
    Anything specific to a particular experiment (Simple or Coattention) should go in the corresponding subclass.
    """
    
    def __init__(self, train_dataset, val_dataset, model, batch_size, num_epochs, num_data_loader_workers=10):
        self._model = model
        self._num_epochs = num_epochs
        self._log_freq = 100  # Steps
        self._test_freq = 1000  # Steps

        self._train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_data_loader_workers)

        # If you want to, you can shuffle the validation dataset and only use a subset of it to speed up debugging
        self._val_dataset_loader = DataLoader(val_dataset, batch_size=2*batch_size, shuffle=False, num_workers=num_data_loader_workers)

        # Use the GPU if it's available.
        self._cuda = torch.cuda.is_available()

        if self._cuda:
            self._model = self._model.cuda()
        
        self.writer = SummaryWriter()

    def _optimize(self, predicted_answers, true_answers):
        """
        This gets implemented in the subclasses. Don't implement this here.
        """
        raise NotImplementedError()

    def accuracy(self, output, target, topk=(1,)):
        """
        Computes the accuracy over the k top predictions for the specified values of k
        """
        with torch.no_grad():
            output = torch.softmax(output, dim=1)
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = torch.topk(output, maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def validate(self, step):
        # TODO. Should return your validation accuracy
        acc = []
        for batch_id, batch_data in enumerate(self._val_dataset_loader):
            image_input = batch_data['image'].cuda() if self._cuda else batch_data['image']
            question_input = batch_data['question'].cuda() if self._cuda else batch_data['question']
            # question_input = batch_data['question_idxs'].cuda() if self._cuda else batch_data['question_idxs']
            predicted_answer = self._model(image_input, question_input)
            ground_truth_answer = batch_data['answer'].cuda() if self._cuda else batch_data['answer']
            values, ground_truth_indices = ground_truth_answer.max(1)

            loss = self._optimize(predicted_answer, ground_truth_indices)

            accu = self.accuracy(predicted_answer, ground_truth_indices)[0]
            acc.append(accu.cpu().numpy())

            validate_step = step * len(self._val_dataset_loader) + batch_id

            # self.writer.add_scalar('validate/loss', loss, validate_step)
            # self.writer.add_scalar('validate/accuracy', acc[0], validate_step)

        return sum(acc)/len(acc)

    def train(self):

        for epoch in range(self._num_epochs):
            
            num_batches = len(self._train_dataset_loader)

            for batch_id, batch_data in enumerate(self._train_dataset_loader):
                self._model.train()  # Set the model to train mode
                current_step = epoch * num_batches + batch_id

                # ============
                # TODO: Run the model and get the ground truth answers that you'll pass to your optimizer
                # This logic should be generic; not specific to either the Simple Baseline or CoAttention.
                image_input = batch_data['image'].cuda() if self._cuda else batch_data['image']
                question_input = batch_data['question'].cuda() if self._cuda else batch_data['question']
                question_indices_input = batch_data['question_idxs'].cuda() if self._cuda else batch_data['question_idxs']
                predicted_answer = self._model(image_input, question_input)
                ground_truth_answer = batch_data['answer'].cuda() if self._cuda else batch_data['answer']
                values, ground_truth_indices = ground_truth_answer.max(1)
                # ============
                # Optimize the model according to the predictions
                
                loss = self._optimize(predicted_answer, ground_truth_indices, train=True)

                acc = self.accuracy(predicted_answer, ground_truth_indices)
                
                n_iter = epoch * num_batches + batch_id

                if current_step % self._log_freq == 0:
                    print("Epoch: {}, Batch {}/{} has loss {} and accuracy {}".format(epoch, batch_id, num_batches, loss, acc[0].cpu().numpy()[0]))
                    # TODO: you probably want to plot something here
                    self.writer.add_scalar('train/loss', loss, n_iter)
                    self.writer.add_scalar('train/accuracy', acc[0], n_iter)
                    for tag, value in self._model.named_parameters():
                        tag = tag.replace('.', '/')
                        self.writer.add_histogram(tag, value.data.cpu().numpy(), n_iter)
                        self.writer.add_histogram(tag+'/grad', value.grad.data.cpu().numpy(), n_iter)

                if (current_step % self._test_freq == 0):
                    self._model.eval()
                    val_accuracy = self.validate(current_step/self._test_freq)
                    print("Epoch: {} has val accuracy {}".format(epoch, val_accuracy))
                    # self.writer.add_scalar('test/loss', loss, n_iter)
                    self.writer.add_scalar('test/accuracy', val_accuracy, n_iter)
                    # TODO: you probably want to plot something here


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count