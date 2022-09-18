from typing import Dict

import jax
import jax.numpy as jnp
import haiku as hk
import optax
import pickle
from tqdm import tqdm

from train.utils_dataset import PNetDataset, RNetDataset, ONetDataset, create_dataloaders
from mtcnn.networks import pnet_encoding_t, pnet_fc_t, pnet_bbx_t
from mtcnn.networks import rnet_encoding_t, rnet_fc_t, rnet_bbx_t, rnet_fll_t
from mtcnn.networks import onet_encoding_t, onet_fc_t, onet_bbx_t, onet_fll_t


pnet_opt = optax.adam(2e-4)
rnet_opt = optax.adam(2e-4)
onet_opt = optax.adam(2e-4)


def pnet_loss(pnet_params: Dict[str, hk.Params], batch: Dict[str, jnp.ndarray]):
    encoding = pnet_encoding_t.apply(pnet_params["encoding"], batch["img"])
    fc_pred = jnp.squeeze(pnet_fc_t.apply(pnet_params["fc"], encoding))
    bbx_pred = jnp.squeeze(pnet_bbx_t.apply(pnet_params["bbx"], encoding))

    fc_loss = -jnp.mean(jnp.log(fc_pred)*batch["fc"])
    bbx_loss = jnp.mean(jnp.mean(jnp.square(bbx_pred - batch["bbx"]), -1), where=batch["fc"].at[:, 0].get() > 0.5)

    loss = fc_loss + 0.5*bbx_loss
    return loss, (fc_loss, bbx_loss)


def rnet_loss(rnet_params: Dict[str, hk.Params], batch: Dict[str, jnp.ndarray]):
    encoding = rnet_encoding_t.apply(rnet_params["encoding"], batch["img"])
    fc_pred = rnet_fc_t.apply(rnet_params["fc"], encoding)
    bbx_pred = rnet_bbx_t.apply(rnet_params["bbx"], encoding)
    fll_pred = rnet_fll_t.apply(rnet_params["fll"], encoding)

    fc_loss = -jnp.mean(jnp.sum(jnp.log(fc_pred)*batch["fc"], axis=-1), where=batch["mask_fc"])
    bbx_loss = jnp.mean(jnp.mean(jnp.square(bbx_pred - batch["bbx"]), -1), where=batch["mask_bbx"])
    fll_loss = jnp.mean(jnp.mean(jnp.square(fll_pred - batch["fll"]), -1), where=batch["mask_fll"])

    loss = fc_loss + 5*bbx_loss + 2*fll_loss
    return loss, (fc_loss, bbx_loss, fll_loss)


def onet_loss(onet_params: Dict[str, hk.Params], batch: Dict[str, jnp.ndarray]):
    encoding = onet_encoding_t.apply(onet_params["encoding"], batch["img"])
    fc_pred = onet_fc_t.apply(onet_params["fc"], encoding)
    bbx_pred = onet_bbx_t.apply(onet_params["bbx"], encoding)
    fll_pred = rnet_fll_t.apply(onet_params["fll"], encoding)

    fc_loss = -jnp.mean(jnp.sum(jnp.log(fc_pred)*batch["fc"], axis=-1), where=batch["mask_fc"])
    bbx_loss = jnp.mean(jnp.mean(jnp.square(bbx_pred - batch["bbx"]), -1), where=batch["mask_bbx"])
    fll_loss = jnp.mean(jnp.mean(jnp.square(fll_pred - batch["fll"]), -1), where=batch["mask_fll"])

    loss = fc_loss + 10*bbx_loss + 10*fll_loss
    return loss, (fc_loss, bbx_loss, fll_loss)


@jax.jit
def ema_update(params, avg_params):
    return optax.incremental_update(params, avg_params, step_size=0.001)


@jax.jit
def pnet_accuracy(pnet_params: Dict[str, hk.Params], batch: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    encoding = pnet_encoding_t.apply(pnet_params["encoding"], batch["img"])
    fc_pred = pnet_fc_t.apply(pnet_params["fc"], encoding)
    fc_pred = fc_pred.at[:, 0, 0, 0].get() > 0.5
    fc_true = batch["fc"].at[:, 0].get() > 0.5
    acc = jnp.mean(jnp.where(fc_pred == fc_true, 1, 0))
    return acc


@jax.jit
def rnet_accuracy(rnet_params: Dict[str, hk.Params], batch: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    encoding = rnet_encoding_t.apply(rnet_params["encoding"], batch["img"])
    fc_pred = rnet_fc_t.apply(rnet_params["fc"], encoding)
    fc_pred = fc_pred.at[:, 0].get() > 0.5
    fc_true = batch["fc"].at[:, 0].get() > 0.5
    acc = jnp.mean(jnp.where(fc_pred == fc_true, 1, 0), where=batch["mask_fc"])
    return acc


@jax.jit
def onet_accuracy(onet_params: Dict[str, hk.Params], batch: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    encoding = onet_encoding_t.apply(onet_params["encoding"], batch["img"])
    fc_pred = onet_fc_t.apply(onet_params["fc"], encoding)
    fc_pred = fc_pred.at[:, 0].get() > 0.5
    fc_true = batch["fc"].at[:, 0].get() > 0.5
    acc = jnp.mean(jnp.where(fc_pred == fc_true, 1, 0), where=batch["mask_fc"])
    return acc


@jax.jit
def pnet_update(pnet_params, opt_state, batch):
    value, grads = jax.value_and_grad(pnet_loss, has_aux=True)(pnet_params, batch)
    updates, opt_state = pnet_opt.update(grads, opt_state)
    new_params = optax.apply_updates(pnet_params, updates)
    return new_params, opt_state, value


@jax.jit
def rnet_update(rnet_params, opt_state, batch):
    value, grads = jax.value_and_grad(rnet_loss, has_aux=True)(rnet_params, batch)
    updates, opt_state = rnet_opt.update(grads, opt_state)
    new_params = optax.apply_updates(rnet_params, updates)
    return new_params, opt_state, value


@jax.jit
def onet_update(onet_params, opt_state, batch):
    value, grads = jax.value_and_grad(onet_loss, has_aux=True)(onet_params, batch)
    updates, opt_state = onet_opt.update(grads, opt_state)
    new_params = optax.apply_updates(onet_params, updates)
    return new_params, opt_state, value


def pnet_train(epochs=20, batch_size=16):
    key = jax.random.PRNGKey(42)
    subkeys = jax.random.split(key, 3)

    pnet_params = {
        "encoding": pnet_encoding_t.init(subkeys[0], jax.random.uniform(subkeys[0], (batch_size, 12, 12, 3))),
        "fc": pnet_fc_t.init(subkeys[1], jax.random.uniform(subkeys[1], (batch_size, 1, 1, 32))),
        "bbx": pnet_bbx_t.init(subkeys[2], jax.random.uniform(subkeys[2], (batch_size, 1, 1, 32)))
    }
    with open("models/pnet/params0", "wb") as params_file:
        pickle.dump(pnet_params, params_file)
    opt_state = pnet_opt.init(params=pnet_params)
    train_dl, val_dl = create_dataloaders(PNetDataset(), batch_size, split=0.8)

    for e in range(epochs):
        print("Epoch ", e+1)
        for batch_num, batch in enumerate(train_dl):
            pnet_params, opt_state, loss = pnet_update(pnet_params, opt_state, batch)
            acc = pnet_accuracy(pnet_params, batch)
            if batch_num % 100 == 0:
                print(f"[{batch_num}/{len(train_dl)}]  Loss : {loss[0]:.5f}  | fc loss : {loss[1][0]:.5f}",
                      f" | bbx loss : {loss[1][1]:.5f}  | acc : {acc*100:.2f}%")
        acc = 0
        for batch in tqdm(val_dl):
            acc += pnet_accuracy(pnet_params, batch)
        print("Val acc: ", acc/len(val_dl))

        with open(f"models/pnet/params{e+1}", "wb") as params_file:
            pickle.dump(pnet_params, params_file)


def rnet_train(epochs=20, batch_size=16):
    key = jax.random.PRNGKey(42)
    subkeys = jax.random.split(key, 4)

    rnet_params = {
        "encoding": rnet_encoding_t.init(subkeys[0], jax.random.uniform(subkeys[0], (batch_size, 24, 24, 3))),
        "fc": rnet_fc_t.init(subkeys[1], jax.random.uniform(subkeys[1], (batch_size, 3*3*64))),
        "bbx": rnet_bbx_t.init(subkeys[2], jax.random.uniform(subkeys[2], (batch_size, 3*3*64))),
        "fll": rnet_fll_t.init(subkeys[3], jax.random.uniform(subkeys[3], (batch_size, 3*3*64)))
    }
    with open("models/rnet/params0", "wb") as params_file:
        pickle.dump(rnet_params, params_file)
    opt_state = rnet_opt.init(params=rnet_params)
    train_dl, val_dl = create_dataloaders(RNetDataset(), batch_size, split=0.8)

    for e in range(epochs):
        print("Epoch ", e+1)
        for batch_num, batch in enumerate(train_dl):
            rnet_params, opt_state, loss = rnet_update(rnet_params, opt_state, batch)
            acc = rnet_accuracy(rnet_params, batch)
            if batch_num % 100 == 0:
                print(f"[{batch_num}/{len(train_dl)}]  Loss : {loss[0]:.5f}  | fc loss : {loss[1][0]:.5f}",
                      f" | bbx loss : {loss[1][1]:.5f}  |  fll loss : {loss[1][2]:.5f}  | acc : {acc*100:.2f}%")
        acc = 0
        for batch in tqdm(val_dl):
            acc += rnet_accuracy(rnet_params, batch)
        print("Val acc: ", acc/len(val_dl))

        with open(f"models/rnet/params{e+1}", "wb") as params_file:
            pickle.dump(rnet_params, params_file)


def onet_train(epochs=20, batch_size=16):
    key = jax.random.PRNGKey(42)
    subkeys = jax.random.split(key, 4)

    onet_params = {
        "encoding": onet_encoding_t.init(subkeys[0], jax.random.uniform(subkeys[0], (batch_size, 48, 48, 3))),
        "fc": onet_fc_t.init(subkeys[1], jax.random.uniform(subkeys[1], (batch_size, 2*2*128))),
        "bbx": onet_bbx_t.init(subkeys[2], jax.random.uniform(subkeys[2], (batch_size, 2*2*128))),
        "fll": onet_fll_t.init(subkeys[3], jax.random.uniform(subkeys[3], (batch_size, 2*2*128)))
    }
    with open("models/onet/params0", "wb") as params_file:
        pickle.dump(onet_params, params_file)
    opt_state = onet_opt.init(params=onet_params)
    train_dl, val_dl = create_dataloaders(ONetDataset(), batch_size, split=0.8)

    for e in range(epochs):
        print("Epoch ", e+1)
        for batch_num, batch in enumerate(train_dl):
            onet_params, opt_state, loss = onet_update(onet_params, opt_state, batch)
            acc = onet_accuracy(onet_params, batch)
            if batch_num % 100 == 0:
                print(f"[{batch_num}/{len(train_dl)}]  Loss : {loss[0]:.5f}  | fc loss : {loss[1][0]:.5f}",
                      f" | bbx loss : {loss[1][1]:.5f}  |  fll loss : {loss[1][2]:.5f}  | acc : {acc*100:.2f}%")
        acc = 0
        for batch in tqdm(val_dl):
            acc += onet_accuracy(onet_params, batch)
        print("Val acc: ", acc/len(val_dl))

        with open(f"models/onet/params{e+1}", "wb") as params_file:
            pickle.dump(onet_params, params_file)
