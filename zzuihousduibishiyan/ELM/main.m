clear all;
close all;
clc;

[TrainingTime,TrainingAccuracy] = elm_train('train9.txt', 1, 100,'sig');

[TestingTime, TestingAccuracy] = elm_predict('test9.txt');

k=1:16;
  plot(k,elm_predict('test9.txt'),k,elm_train('train9.txt', 1, 100,'sig'),'.');
  grid on;