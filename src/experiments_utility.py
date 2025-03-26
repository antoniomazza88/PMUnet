from tqdm import tqdm
from network import *
from metrics import *
import os
import imageio.v3 as imio
from torch.utils.data import DataLoader
import torch.optim as optim

from dataloader import S5P_P25

def get_test_list(args, suffix='test'):
    return list(np.load(os.path.join(args.dataset_path, "{}.npy".format(suffix.capitalize()))))

def save_checkpoint(net, out_dir):
    # Save checkpoint
    state = {
        'net': net.state_dict()
    }
    checkpoint_dir = os.path.join(out_dir, "checkpoint/")
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = 'model_ckpt.t7'
    print('Saving ', filename)
    torch.save(state, os.path.join(checkpoint_dir, filename))
    return checkpoint_dir


def load_checkpoint(net, path):
    if path is not None and not path.endswith('.t7'):
        checkpoint_dir = path
        ckpts = os.listdir(checkpoint_dir)
        ckpts = [x for x in ckpts if x.__contains__('ckpt')]
        ckpts.sort()
        filename = os.path.join(checkpoint_dir, ckpts[-1])
    else:
        filename = path
    print("Loading ", filename)
    checkpoint = torch.load(filename)
    net.load_state_dict(checkpoint["net"])
    return net


def trainloop(args):
    device = init_device()
    net, optimizer, expname, out_dir = init_experiment(args, device)
    trainloader, validloader, test_load, gen_load = init_dataloader(args, device)

    val_tot = dict()
    min_loss = torch.Tensor([1e10])
    times = []
    cont = 0
    checkpoint_dir = save_checkpoint(net, out_dir)
    stop = False
    while not stop:
        if validloader is not None:
            val_losses = val_epoch(net, validloader, device)
            for k in val_losses.keys():
                if not k in val_tot.keys():
                    val_tot[k] = torch.zeros((1, 501))
                val_tot[k][:, cont] = val_losses[k]
                torch.save(val_tot[k], os.path.join(checkpoint_dir, 'val_{}.t7'.format(k)))
            if val_losses['loss'] < min_loss:
                min_loss = val_losses['loss']
                checkpoint_dir = save_checkpoint(net, out_dir)

            cont = cont + 1
        stop = train_epoch(optimizer, cont, expname, net, trainloader, device)
        stop = stop or cont >= 500
    # checkpoint_dir = save_checkpoint(net, out_dir)
    val_losses = val_epoch(net=net, dataloader=validloader, device=device)
    for k in val_losses.keys():
        val_tot[k][:, cont] = val_losses[k]
        torch.save(val_tot[k], os.path.join(checkpoint_dir, 'val_{}.t7'.format(k)))
    test_phase(net, test_load, out_dir, get_test_list(args, suffix = 'test'), suffix = 'test')
    test_phase(net, gen_load, out_dir, get_test_list(args, suffix='gen'), suffix='gen')
    return net


def init_experiment(args, device):

    if args.model == 'PMSlim':
        model = PMSlim()
    elif args.model == 'PMUnet':
        model = PMUnet()
    elif args.model == 'PMRes':
        model = PMRes()

    model = model.float().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    if args.exp_name == None:
        name = "{}".format(args.model)
    else:
        name = args.exp_name

    save = os.path.join(args.out_folder, name)
    os.makedirs(save, exist_ok=True)

    return model, optimizer, name, save


def init_dataloader(args, device):
    list_files = np.load(os.path.join(args.dataset_path, "Train.npy"))
    val_files = np.load(os.path.join(args.dataset_path, "Val.npy"))
    test_files = np.load(os.path.join(args.dataset_path, "Test.npy"))
    gen_files = np.load(os.path.join(args.dataset_path, "Gen.npy"))
    train_data = S5P_P25(list_files, device, transform=True)
    val_data = S5P_P25(val_files, device)
    test_data = S5P_P25(test_files, device)
    gen_data = S5P_P25(gen_files, device)
    train_load = DataLoader(train_data, batch_size=4, shuffle=True)
    val_load = DataLoader(val_data, batch_size=1, shuffle=False)
    test_load = DataLoader(test_data, batch_size=1, shuffle=False)
    gen_load = DataLoader(gen_data, batch_size=1, shuffle=False)

    return train_load, val_load, test_load, gen_load

def init_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = 'cpu'
    return device


def train_epoch(optimizer, epoch, expname, net, dataloader, device):

    print('\nTraining Epoch: %d, Learning rate: %f, Expdir %s' % (epoch, 1e-5, expname))
    net.train()

    stats_num = dict()
    stats_cum = dict()

    loss = PMLoss(device=device).to(device)
    with tqdm(dataloader, unit="batch") as pbar:

        for input, label in pbar:
            optimizer.zero_grad()
            pred = net(input)
            stats_one = dict()
            l1 = loss(pred, label)
            stats_one["loss"] = l1.item()
            l1 = l1
            l1.backward()
            optimizer.step()

            with torch.no_grad():
                stats_one = dict()
                stats_one["loss"] = l1.item()

            pbar.set_postfix(ordered_dict=stats_one)

            for stats_key in stats_one:
                if not stats_key in stats_cum.keys():
                    stats_cum[stats_key] = 0
                    stats_num[stats_key] = 0
                stats_cum[stats_key] = stats_cum[stats_key] + stats_one[stats_key]
                stats_num[stats_key] = stats_num[stats_key] + 1

        for stats_key in stats_one:
            stats_cum[stats_key] = stats_cum[stats_key] / stats_num[stats_key]

        print('Epoch: %d |' % epoch, end='')
        for stats_key, stats_value in stats_cum.items():
            print(' %5s: %.5f | ' % (stats_key, stats_value), end='\n')
        print("")

        return False

def val_epoch(net, dataloader, device):
    stats_cum = dict()
    stats_num = dict()
    net.eval()
    with torch.no_grad():
        loss = PMLoss(device=device).to(device)
        print("\nValidation ")
        with tqdm(dataloader, unit="batch") as pbar:
            cont = 0
            for input, label in pbar:
                torch.cuda.empty_cache()

                pred = net(input)
                stats_one = dict()
                l1 = loss(pred, label)
                stats_one["loss"] = l1.item()
                pbar.set_postfix(ordered_dict=stats_one)
                cont = cont + 1
                for stats_key in stats_one:
                    if not stats_key in stats_cum.keys():
                        stats_cum[stats_key] = 0
                        stats_num[stats_key] = 0
                    stats_cum[stats_key] = stats_cum[stats_key] + stats_one[stats_key]
                    stats_num[stats_key] = stats_num[stats_key] + 1

            for stats_key in stats_one:
                stats_cum[stats_key] = stats_cum[stats_key] / stats_num[stats_key]

            for stats_key, stats_value in stats_cum.items():
                print('')
                print(' %5s: %.5f | ' % (stats_key, stats_value), end='')
            print("")
    return stats_cum

def test_phase(net, dataloader, outdir, list_patches=[], suffix = 'test'):
    net.eval()
    perf = dict()
    with torch.no_grad():
        with tqdm(dataloader, unit="batch") as pbar:
            k = 0
            for input, label in pbar:

                if len(list_patches) > 0:
                    print("Testing on ", list_patches[k])
                    name = os.path.split(list_patches[k])[-1].split('.')[0]
                    pred_name = 'pred_{}.tif'.format(name)
                    orig_name = 'orig_{}.tif'.format(name)
                else:
                    pred_name = 'pred.tif'
                    orig_name = 'orig.tif'
                output_dirname = os.path.join(outdir, suffix)
                tif_filename = os.path.join(output_dirname, pred_name)
                orig_filename = os.path.join(output_dirname, orig_name)
                os.makedirs(output_dirname, exist_ok=True)
                pred = net(input)
                p = pred.to('cpu')
                o = np.squeeze(label.cpu().numpy())
                o = crop_img(o, name)
                p = np.squeeze(p.numpy()[0, :, :])
                p = crop_img(p, name)
                perf[name] = eval_perf(o,p)
                imio.imwrite(tif_filename, p)
                imio.imwrite(orig_filename, o)
                k = k + 1

        perf = eval_overall(perf)
        print(perf['overall'])
        json_object = json.dumps(perf, indent=4)
        with open(os.path.join(outdir, "{}_perf.json".format(suffix)), "w") as outfile:
            outfile.write(json_object)


def init_test(args, device):
    if args.model == 'PMSlim':
        model = PMSlim()
    elif args.model == 'PMUnet':
        model = PMUnet()
    elif args.model == 'PMRes':
        model = PMRes()


    if args.exp_name == None:
        name = "{}".format(args.model)
    else:
        name = args.exp_name
    save = os.path.join(args.out_folder, name)
    os.makedirs(save, exist_ok=True)
    model = load_checkpoint(model, os.path.join(save, 'checkpoint'))
    model = model.float().to(device)
    return model, save


def init_testloader(args, device):
    test_files = np.load(os.path.join(args.dataset_path, "Test.npy"))
    test_data = S5P_P25(test_files, device)
    test_load = DataLoader(test_data, batch_size=1, shuffle=False)
    gen_files = np.load(os.path.join(args.dataset_path, "Gen.npy"))
    gen_data = S5P_P25(gen_files, device)
    gen_load = DataLoader(gen_data, batch_size=1, shuffle=False)

    return test_load, list(test_files), gen_load, list(gen_files)

def testing(args):
    device = init_device()
    net, out_dir = init_test(args, device)
    test_load, test_list, gen_load, gen_list = init_testloader(args, device)
    test_phase(net, test_load, out_dir, test_list, suffix = 'test')
    test_phase(net, gen_load, out_dir, gen_list, suffix = 'gen')

def crop_img(img, acq):
    if acq.__contains__('20200102'):
        return img[:110, :110]
    if acq.__contains__('20200112'):
        return img[:50, 27:77]
    elif acq.__contains__('20200123'):
        return img[:60, :60]
    elif acq.__contains__('20200416'):
        return img[20:60, 40:80]
    elif acq.__contains__('20200524'):
        return img[:66, 66:132]
    elif acq.__contains__('20200813'):
        return img[:50, :50]
    elif acq.__contains__('20201021'):
        return img[:58, 29:87]
    elif acq.__contains__('20210225'):
        return img[-67:, 80:147]
    elif acq.__contains__('20210729'):
        return img[-80:, -80:]
    elif acq.__contains__('20200207'):
        return img[:60, :60]
    elif acq.__contains__('20210905'):
        return img[:40, 20:60]
    # else:
    #     return img



