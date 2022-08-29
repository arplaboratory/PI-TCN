#!/usr/bin/env python
import rospy
import bagpy
import pandas as pd
import numpy as np
import glob
import random
import os
import sys
import shutil
import itertools
from tqdm import tqdm
from time import strftime, gmtime
import scipy
from scipy.signal import butter, lfilter, lfilter_zi, freqz
from scipy.spatial.transform import Rotation as scipy_rotation

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.widgets import Slider

plt.rcParams["figure.figsize"] = (19.20, 10.80)
font = {"family" : "sans",
        "weight" : "normal",
        "size"   : 28}
matplotlib.rc("font", **font)
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
colors = ["#7d7376","#365282","#e84c53","#edb120"]

def initFolders():
  try:
    shutil.rmtree("info")
    shutil.rmtree("train")
    shutil.rmtree("test")
  except OSError as e:
    pass
  os.mkdir("info")
  os.mkdir("train")
  os.mkdir("test")

def parseBag(topic, path):
  bag = bagpy.bagreader(path, verbose=False)
  return pd.read_csv(bag.message_by_topic(topic))

def shift(arr):
  if len(arr.shape) == 1:
    arr = arr.reshape((len(arr), 1))
  return np.vstack((np.nan * np.ones(shape=(1, arr.shape[1])), arr[:-1]))

def shiftFilteredSpline(sampled_data, step):
  for k in sampled_data.keys():
    if "filt" in k:
      sampled_data[k] = sampled_data[k][step:]
    else:
      sampled_data[k] = sampled_data[k][:-step]

def differentiate(arr, dt):
  a_dot = (arr - shift(arr)) / dt
  a_dot[0, :] = np.zeros((a_dot.shape[1]))
  return a_dot

def differentiateFivePointStencil(arr, dt):
  a_dot = [
    [np.zeros((arr.shape[1]))],
    [(arr[2] - arr[1]) / dt[1]]
  ]
  for i in range(2, len(arr) - 2):
    a_dot.append([(- arr[i + 2] + 8 * arr[i + 1] - 8 * arr[i - 1] + arr[i - 2]) * (1. / (12 * dt[i]))])
  a_dot.append([(arr[-2] - arr[-3]) / dt[-2]])
  a_dot.append([(arr[-1] - arr[-2]) / dt[-1]])
  return np.concatenate(a_dot, axis=0)

def dropNoise(arr, t, dt):
  drop_indeces = []
  for i in range(1, len(arr)):
    if np.any(abs(arr[i]) - abs(sum(arr[i-4:i])) > 0):
      drop_indeces.append(i)
  return np.delete(arr, drop_indeces, axis=0), np.delete(t, drop_indeces, axis=0), np.delete(dt, drop_indeces, axis=0)

def applySavitzkyGolayFilter(arr, window_length, poly_order):
  return np.array([savitzkyGolayFilter(arr[:, i], window_length, poly_order) for i in range(arr.shape[1])]).T

def savitzkyGolayFilter(data, window_length, poly_order):
  return scipy.signal.savgol_filter(data, window_length, poly_order)

def applyButterLowpassFilter(arr):
  return np.array([butterLowpassFilter(arr[:, i]) for i in range(arr.shape[1])]).T

def butterLowpassFilter(data):
  nyq = 0.5 * frequency
  normal_cutoff = cutoff / nyq
  b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)
  zi = lfilter_zi(b, a)
  y = lfilter(b, a, data, zi=data[0] * zi)[0]
  return y

def appendHistory(df, data_columns, label_columns, history_length):
  state_columns = [col for col in data_columns if "f" not in col]
  df_state = df[state_columns]
  df_state_history = df_state.rename(columns={col: col + "_t" for col in state_columns})
  for j in range(1, 1 + history_length):
    shifted_df = df_state.shift(j)
    for k in range(j):
      shifted_df.iloc[k] = shifted_df.iloc[j]  # repeat initial elements where shift has left NaNs
    col_names = {col: col + "_t-" + str(j) for col in list(df.columns)}
    shifted_df.rename(columns=col_names, inplace=True)
    df_state_history = pd.concat([shifted_df, df_state_history], axis=1)

  input_columns = [col for col in data_columns if "f" in col]
  df_input = df[input_columns]
  df_input_history = df_input.rename(columns={col: col + "_t" for col in input_columns})
  for j in range(1, 1 + history_length):
    shifted_df = df_input.shift(j)
    for k in range(j):
      shifted_df.iloc[k] = shifted_df.iloc[j]  # repeat initial elements where shift has left NaNs
    col_names = {col: col + "_t-" + str(j) for col in list(df.columns)}
    shifted_df.rename(columns=col_names, inplace=True)
    df_input_history = pd.concat([shifted_df, df_input_history], axis=1)

  df_history = pd.concat([df_state_history, df_input_history, df[label_columns]], axis=1)
  return df_history

def computeSpline(str, arr, t, steps, cols):
  if len(t.shape) == 1:
    t = t.reshape((t.shape[0], 1))
  splines = {}
  for i in range(len(cols)):
    splines[str + "_" + cols[i]] = scipy.interpolate.CubicSpline(t[:, 0], arr[:, i])(steps, 0)
  return splines

def saveTrajectory(df, file_name):
  df_splitted = np.array_split(df, 1)
  # df_splitted = np.array_split(df, int(df["t"].iloc[-1] / 10.) + 1)
  for i, df_slice in enumerate(df_splitted):
    pp = PdfPages("info/" + file_name + "_" + str(i) + ".pdf")

    fig, axes = plt.subplots(nrows=3, ncols=1)
    fig.suptitle('Position', x=0.5, y=0.95)
    axes[0].plot(df_slice["t"], df_slice["p_x"], c=colors[0])
    axes[1].plot(df_slice["t"], df_slice["p_y"], c=colors[0])
    axes[2].plot(df_slice["t"], df_slice["p_z"], c=colors[0])
    axes[0].set_ylabel(r'$p_x$ $[m/s]$')
    axes[1].set_ylabel(r'$p_y$ $[m/s]$')
    axes[2].set_ylabel(r'$p_z$ $[m/s]$')
    axes[2].set_xlabel(r'Time $[s]$')
    axes[0].grid(); axes[1].grid(); axes[2].grid()
    fig.align_ylabels(axes)
    pp.savefig(fig)

    fig, axes = plt.subplots(nrows=3, ncols=1)
    fig.suptitle('Linear Velocity', x=0.5, y=0.95)
    axes[0].plot(df_slice["t"], df_slice["v_x"], c=colors[1])
    axes[1].plot(df_slice["t"], df_slice["v_y"], c=colors[1])
    axes[2].plot(df_slice["t"], df_slice["v_z"], c=colors[1])
    axes[0].set_ylabel(r'$v_x$ $[m/s]$')
    axes[1].set_ylabel(r'$v_y$ $[m/s]$')
    axes[2].set_ylabel(r'$v_z$ $[m/s]$')
    axes[2].set_xlabel(r'Time $[s]$')
    axes[0].grid(); axes[1].grid(); axes[2].grid()
    fig.align_ylabels(axes)
    pp.savefig(fig)

    fig, axes = plt.subplots(nrows=3, ncols=1)
    fig.suptitle('Linear Acceleration', x=0.5, y=0.95)
    axes[0].plot(df_slice["t"], df_slice["vdot_x"], c=colors[1], label="Label")
    # axes[0].plot(df_slice["t"], df_slice["vdot_nom_x"], c=colors[2], label="Nominal")
    axes[1].plot(df_slice["t"], df_slice["vdot_y"], c=colors[1])
    # axes[1].plot(df_slice["t"], df_slice["vdot_nom_y"], c=colors[2])
    axes[2].plot(df_slice["t"], df_slice["vdot_z"], c=colors[1])
    # axes[2].plot(df_slice["t"], df_slice["vdot_nom_z"], c=colors[2])
    axes[0].set_ylabel(r'$\dot{v}_x$ $[m/s]$')
    axes[1].set_ylabel(r'$\dot{v}_y$ $[m/s]$')
    axes[2].set_ylabel(r'$\dot{v}_z$ $[m/s]$')
    axes[2].set_xlabel(r'Time $[s]$')
    axes[0].grid(); axes[1].grid(); axes[2].grid()
    fig.align_ylabels(axes)
    # axes[0].legend(loc='upper center', bbox_to_anchor=(0.7, 1.5), ncol=3)
    pp.savefig(fig)

    fig, axes = plt.subplots(nrows=4, ncols=1)
    fig.suptitle('Quaternion', x=0.5, y=0.95)
    axes[0].plot(df_slice["t"], df_slice["q_w"], c=colors[1])
    axes[1].plot(df_slice["t"], df_slice["q_x"], c=colors[1])
    axes[2].plot(df_slice["t"], df_slice["q_y"], c=colors[1])
    axes[3].plot(df_slice["t"], df_slice["q_z"], c=colors[1])
    axes[0].set_ylabel(r'$q_w$ $[m/s]$')
    axes[1].set_ylabel(r'$q_x$ $[m/s]$')
    axes[2].set_ylabel(r'$q_y$ $[m/s]$')
    axes[3].set_ylabel(r'$q_z$ $[m/s]$')
    axes[3].set_xlabel(r'Time $[s]$')
    axes[0].grid(); axes[1].grid(); axes[2].grid(); axes[3].grid()
    fig.align_ylabels(axes)
    pp.savefig(fig)

    fig, axes = plt.subplots(nrows=3, ncols=1)
    fig.suptitle('Angular Velocity', x=0.5, y=0.95)
    axes[0].plot(df_slice["t"], df_slice["w_x"], c=colors[1])
    axes[1].plot(df_slice["t"], df_slice["w_y"], c=colors[1])
    axes[2].plot(df_slice["t"], df_slice["w_z"], c=colors[1])
    axes[0].set_ylabel(r'$w_x$ $[rad/s]$')
    axes[1].set_ylabel(r'$w_y$ $[rad/s]$')
    axes[2].set_ylabel(r'$w_z$ $[rad/s]$')
    axes[2].set_xlabel(r'Time $[s]$')
    axes[0].grid(); axes[1].grid(); axes[2].grid()
    fig.align_ylabels(axes)
    pp.savefig(fig)

    fig, axes = plt.subplots(nrows=3, ncols=1)
    fig.suptitle('Angular Acceleration', x=0.5, y=0.95)
    axes[0].plot(df_slice["t"], df_slice["wdot_x"], c=colors[1], label="Label")
    # axes[0].plot(df_slice["t"], df_slice["wdot_nom_x"], c=colors[2], label="Nominal")
    axes[1].plot(df_slice["t"], df_slice["wdot_y"], c=colors[1])
    # axes[1].plot(df_slice["t"], df_slice["wdot_nom_y"], c=colors[2])
    axes[2].plot(df_slice["t"], df_slice["wdot_z"], c=colors[1])
    # axes[2].plot(df_slice["t"], df_slice["wdot_nom_z"], c=colors[2])
    axes[0].set_ylabel(r'$\dot{w}_x$ $[rad/s^2]$')
    axes[1].set_ylabel(r'$\dot{w}_y$ $[rad/s^2]$')
    axes[2].set_ylabel(r'$\dot{w}_z$ $[rad/s^2]$')
    axes[2].set_xlabel(r'Time $[s]$')
    axes[0].grid(); axes[1].grid(); axes[2].grid()
    fig.align_ylabels(axes)
    # axes[0].legend(loc='upper center', bbox_to_anchor=(0.7, 1.5), ncol=2)
    pp.savefig(fig)

    fig, axes = plt.subplots(nrows=4, ncols=1)
    fig.suptitle('Motor Thrust', x=0.5, y=0.95)
    axes[0].plot(df_slice["t"], df_slice["f_0"], c=colors[1])
    axes[1].plot(df_slice["t"], df_slice["f_1"], c=colors[1])
    axes[2].plot(df_slice["t"], df_slice["f_2"], c=colors[1])
    axes[3].plot(df_slice["t"], df_slice["f_3"], c=colors[1])
    axes[0].set_ylabel(r'$f_0$ $[N]$')
    axes[1].set_ylabel(r'$f_1$ $[N]$')
    axes[2].set_ylabel(r'$f_2$ $[N]$')
    axes[3].set_ylabel(r'$f_3$ $[N]$')
    axes[3].set_xlabel(r'Time $[s]$')
    axes[0].grid(); axes[1].grid(); axes[2].grid(); axes[3].grid()
    fig.align_ylabels(axes)
    pp.savefig(fig)

    fig = plt.figure()
    plt.title('Position X-Y')
    plt.plot(df_slice["p_x"], df_slice["p_y"], c=colors[0])
    plt.xlabel(r'$p_x$ $[m]$')
    plt.ylabel(r'$p_y$ $[m]$')
    plt.grid()
    pp.savefig(fig)

    fig = plt.figure()
    plt.title('Position X-Z')
    plt.plot(df_slice["p_x"], df_slice["p_z"], c=colors[0])
    plt.xlabel(r'$p_x$ $[m]$')
    plt.ylabel(r'$p_z$ $[m]$')
    plt.grid()
    pp.savefig(fig)

    fig = plt.figure()
    plt.title('Position Y-Z')
    plt.plot(df_slice["p_y"], df_slice["p_z"], c=colors[0])
    plt.xlabel(r'$p_y$ $[m]$')
    plt.ylabel(r'$p_z$ $[m]$')
    plt.grid()
    fig.align_ylabels(axes)
    pp.savefig(fig)

    pp.close()
    plt.close('all')

def nominalModel(data, thrust_coeff, torque_coeff, inertia, mass, arm_length):
  q = np.vstack((np.vstack((data["q_w"], data["q_x"])), np.vstack((data["q_y"], data["q_z"])))).T
  f = np.vstack((np.vstack((data["f_0"], data["f_1"])), np.vstack((data["f_2"], data["f_3"])))).T
  w = np.vstack((np.vstack((data["w_x"], data["w_y"])), data["w_z"])).T

  vdot = []
  for i in range(q.shape[0]):
    thrust = f[i, 0] + f[i, 1] + f[i, 2] + f[i, 3]
    quat_norm = q[i, 0] ** 2 + q[i, 1] ** 2 + q[i, 2] ** 2 + q[i, 3] ** 2
    vdot.append([
      (1. / mass) * thrust * 2. * (q[i, 0] * q[i, 2] + q[i, 1] * q[i, 3]) / quat_norm,
      (1. / mass) * thrust * 2. * (q[i, 2] * q[i, 3] - q[i, 0] * q[i, 1]) / quat_norm,
      (1. / mass) * thrust * (1. - 2. * q[i, 1] * q[i, 1] - 2. * q[i, 2] * q[i, 2]) / quat_norm - 9.8066
    ])

  wdot = []
  km_kf = torque_coeff / thrust_coeff
  for i in range(w.shape[0]):
    wdot.append([
      (arm_length * (f[i, 0] + f[i, 1] - f[i, 2] - f[i, 3])  + inertia[1] * w[i, 1] * w[i, 2] - inertia[2] * w[i, 1] * w[i, 2]) / inertia[0],
      (arm_length * (-f[i, 0] + f[i, 1] + f[i, 2] - f[i, 3]) - inertia[0] * w[i, 0] * w[i, 2] + inertia[2] * w[i, 0] * w[i, 2]) / inertia[1],
      (km_kf      * (f[i, 0] - f[i, 1] + f[i, 2] - f[i, 3])  + inertia[0] * w[i, 0] * w[i, 1] - inertia[1] * w[i, 0] * w[i, 1]) / inertia[2]
    ])

  return np.array(vdot), np.array(wdot)

def processBag(path):
  # quadrotor physics
  mass = 0.25
  thrust_coeff = 4.37900e-09
  torque_coeff = 3.97005e-11
  arm_length = 0.076
  inertia = np.array([0.000601, 0.000589, 0.001076])

  # low pass filter
  frequency = 100.0
  cutoff = 5
  order = 4
  shift_step = 9

  # strong noise filter (manual)
  drop_noise_thresh = 1
  drop_noise_idx = 2

  # load dataframes
  df_odom = parseBag('/dragonfly17/odom', path)
  df_imu = parseBag('/dragonfly17/imu', path)
  df_motor = parseBag('/dragonfly17/motor_rpm', path)

  # compute time
  t_odom = df_odom.apply(lambda r: rospy.Time(r["header.stamp.secs"], r["header.stamp.nsecs"]).to_sec(), axis=1)
  t_imu = df_imu.apply(lambda r: (r["header.stamp.secs"] + (r["header.stamp.nsecs"] / 1e9)), axis=1)
  t_motor = df_motor.apply(lambda r: rospy.Time(r["header.stamp.secs"], r["header.stamp.nsecs"]).to_sec(), axis=1)
  t_odom = t_odom.to_numpy().reshape((len(t_odom), 1))
  t_imu = t_imu.to_numpy().reshape((len(t_imu), 1))
  t_motor = t_motor.to_numpy().reshape((len(t_motor), 1))
  dt_odom = t_odom - shift(t_odom)
  dt_imu = t_imu - shift(t_imu)
  dt_motor = t_motor - shift(t_motor)
  dt_odom[np.isnan(dt_odom)] = 0.
  dt_imu[np.isnan(dt_imu)] = 0.
  dt_motor[np.isnan(dt_motor)] = 0.

  # sampling steps
  sampling_bounds = [max(np.min(t_odom), np.min(t_imu), np.min(t_motor)),
                     min(np.max(t_odom), np.max(t_imu), np.max(t_motor))]
  sampling_bounds[0] = round(sampling_bounds[0] - sampling_bounds[0] % 1. / frequency, 4)
  sampling_bounds[1] = round(sampling_bounds[1] - sampling_bounds[1] % 1. / frequency, 4)
  sampling_steps = np.arange(sampling_bounds[0], sampling_bounds[1], 1. / frequency)[:-1]

  # store all processed data in a dictionary
  sampled_data = {"t": sampling_steps}

  ## position
  p = df_odom[["pose.pose.position.x", "pose.pose.position.y", "pose.pose.position.z"]].to_numpy()
  sampled_data.update(computeSpline("p", p, t_odom, sampling_steps, "xyz"))

  ## orientation
  q = df_odom[["pose.pose.orientation.w", "pose.pose.orientation.x",
               "pose.pose.orientation.y", "pose.pose.orientation.z"]].to_numpy()
  # r = [scipy_rotation.from_quat(q[i, :]).as_matrix() for i in range(q.shape[0])]
  # q = applyButterLowpassFilter(q)
  sampled_data.update(computeSpline("q", q, t_odom, sampling_steps, "wxyz"))

  ## motor speeds
  u = df_motor[["rpm_0", "rpm_1", "rpm_2", "rpm_3"]].to_numpy()
  u, t_filt, _ = dropNoise(u, t_motor, dt_motor)
  # u = applyButterLowpassFilter(u)
  sampled_data.update(computeSpline("u", u, t_filt, sampling_steps, "0123"))

  ## motor thrusts
  f = (u ** 2) * thrust_coeff
  sampled_data.update(computeSpline("f", f, t_filt, sampling_steps, "0123"))

  ## linear velocity
  v = df_odom[["twist.twist.linear.x", "twist.twist.linear.y", "twist.twist.linear.z"]].to_numpy()
  v, t_filt, dt_filt = dropNoise(v, t_odom, dt_odom)
  # v = applyButterLowpassFilter(v)
  # v = applySavitzkyGolayFilter(v, window_length=101, poly_order=4)
  sampled_data.update(computeSpline("v", v, t_filt, sampling_steps, "xyz"))

  ## angular velocity
  # w = df_odom[["twist.twist.angular.x", "twist.twist.angular.y", "twist.twist.angular.z"]].to_numpy()
  w = df_imu[["angular_velocity.x", "angular_velocity.y", "angular_velocity.z"]].to_numpy()
  w = w * np.array([1, -1, -1])
  # w = applyButterLowpassFilter(w)
  # w = applySavitzkyGolayFilter(w, window_length=101, poly_order=4)
  sampled_data.update(computeSpline("w", w, t_imu, sampling_steps, "xyz"))

  ## linear acceleration
  vdot = differentiate(v, dt_filt)
  # vdot = differentiateFivePointStencil(v, dt_filt)
  # vdot = applyButterLowpassFilter(vdot)
  # vdot = applySavitzkyGolayFilter(vdot, window_length=101, poly_order=4)
  sampled_data.update(computeSpline("vdot", vdot, t_filt, sampling_steps, "xyz"))

  ## angular acceleration
  wdot = differentiate(w, dt_imu)
  # wdot = differentiateFivePointStencil(w, dt_imu)
  # wdot = applyButterLowpassFilter(wdot)
  # wdot = applySavitzkyGolayFilter(wdot, window_length=101, poly_order=4)
  sampled_data.update(computeSpline("wdot", wdot, t_imu, sampling_steps, "xyz"))

  ## nominal model
  vdot_nom, wdot_nom = nominalModel(sampled_data, thrust_coeff, torque_coeff, inertia, mass, arm_length)
  sampled_data.update(computeSpline("vdot_nom", vdot_nom, sampled_data["t"], sampling_steps, "xyz"))
  sampled_data.update(computeSpline("wdot_nom", wdot_nom, sampled_data["t"], sampling_steps, "xyz"))

  # shift filtered data
  shiftFilteredSpline(sampled_data, step=shift_step)

  # shift time so that it starts from 0.0
  sampled_data["t"] -= sampled_data["t"][0]

  return sampled_data

def main():
  # select number of testing trajectories to keep out the training set
  num_test_traj = 1

  # history length (minimum is 1)
  history_length = 1

  # data columns
  data_columns = ["v_x", "v_y", "v_z", "q_w", "q_x", "q_y", "q_z",
                  "w_x", "w_y", "w_z", "f_0", "f_1", "f_2", "f_3"]
  label_columns = ["vdot_x", "vdot_y", "vdot_z", "wdot_x", "wdot_y", "wdot_z"]
  columns = list(itertools.chain(data_columns, label_columns))

  # create dataset folders
  initFolders()

  # find all bags
  bag_paths = glob.glob("bags/*.bag")
  # random.shuffle(bag_paths)

  # process each bag
  processed_data = [processBag(path) for path in tqdm(bag_paths, desc="Preprocessing")]

  # create dataset from processed bags
  for i, path in tqdm(enumerate(bag_paths), total=len(bag_paths), desc="Saving"):
    file_name = path.split("/")[-1][:-4]
    shutil.rmtree("bags/" + file_name)

    # save info about preprocessed trajectories
    df = pd.concat({k: pd.Series(v) for k, v in processed_data[i].items()}, axis=1)
    saveTrajectory(df, file_name)

    # append history to dataframe and save it
    data_df = appendHistory(df, data_columns, label_columns, history_length)
    # nom_df = appendHistory(df, data_columns, [c.replace("filt", "nom") for c in label_columns], history_length)
    if i < num_test_traj:
      data_df.to_csv("test/" + file_name + ".csv", index=False)
    elif i == num_test_traj:
      data_df.to_csv("train/data.csv", index=False)
      # nom_df.to_csv("train/nominal.csv", index=False)
    else:
      data_df.to_csv("train/data.csv", index=False, mode='a', header=False)
      # nom_df.to_csv("train/nominal.csv", index=False, mode='a', header=False)

if __name__ == "__main__":
  main()
