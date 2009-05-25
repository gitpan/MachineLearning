#!/usr/bin/perl

# nn_test.t:  Test the MachineLearning::NeuralNetwork module.
use strict;
use utf8;
use English;
use Test::More ("tests" => 45);
use MachineLearning::NeuralNetwork;

# Test the constructor and the accessor methods:
my $nn = MachineLearning::NeuralNetwork->new({
  "Name" => "Sample Neural Network",
  "Description" => "Sample neural network",
  "HiddenLayerSize"  => 13,
  "InputFieldNames"  => ["First Serve Accuracy", "Return Speed",
                         "Errors Per Set", "Height", "Weight"],
  "OutputFieldNames" => ["Result"]});
ok($nn->get_success(), "Successful object creation");
is($nn->get_message(), "", "Empty error message, as expected");
is($nn->get_name(), "Sample Neural Network", "Correct name");
like($nn->get_description(), qr/network\z/is, "Correct description");
isa_ok($nn, "MachineLearning::NeuralNetwork");
can_ok($nn, $_) for ("run", "train", "test", "grow", "prune", "save");
ok(scalar(@{ $nn->{"_input_neurons"} }) == 5, "Correct input layer size");
ok(scalar(@{ $nn->{"_hidden_neurons"} }) == 13, "Correct hidden layer size");
ok(scalar(@{ $nn->{"_output_neuron"} }) == 1, "Correct output layer size");
is($nn->{"_input_neurons"}->[0]->{"Name"}, "First Serve Accuracy",
  "Correct first input neuron name");
is($nn->{"_output_neuron"}->[0]->{"Name"}, "Result",
  "Correct output neuron name");

# Test the train() method:
my $trained_ok = $nn->train({
  "TrainingDataFilePath" => "sample_data_nn.csv",
  "CyclesPerTemperature" => 2,
  "MinimumOnesPercentage" => 5.0});

ok($trained_ok, "Trained OK");
ok(scalar(@{ $nn->{"_input_neurons"}->[0]->{"Weights"} }),
  "Input neurons have weights after training");
ok(scalar(@{ $nn->{"_hidden_neurons"}->[0]->{"Weights"} }),
  "Hidden neurons have weights after training");
ok($nn->{"_hidden_neurons"}->[0]->{"Threshold"} > 0,
  "Hidden neurons have thresholds after training");
ok($nn->{"_output_neuron"}->[0]->{"Threshold"} > 0,
  "Output neuron has a threshold after training");

# Test the test() method:
my $report = $nn->test("sample_data_nn.csv");

ok(length $report, "Test completed correctly");
print $report;
print "____________\n";

# Test the run() method:
ok($nn->run("sample_data_nn_run.csv"), "Ran OK");

# Test the grow() method:
my $new_hidden_layer_size = $nn->grow({
  "TrainingDataFilePath" => "sample_data_nn.csv",
  "ValidationDataFilePath" => "sample_data_nn.csv",
  "CyclesPerTemperature" => 2,
  "MinimumOnesPercentage" => 5.0});

ok($new_hidden_layer_size > 0, "Grew fine");
print "New hidden layer size:  $new_hidden_layer_size\n";

# Test the prune() method:
my $grown_hidden_layer_size = scalar(@{ $nn->{"_hidden_neurons"} });
my $pruned_hidden_layer_size = $nn->prune({
  "TrainingDataFilePath" => "sample_data_nn.csv",
  "ValidationDataFilePath" => "sample_data_nn.csv",
  "CyclesPerTemperature" => 2,
  "MinimumOnesPercentage" => 5.0});

ok($pruned_hidden_layer_size > 0, "Pruned fine");
ok($pruned_hidden_layer_size <= $grown_hidden_layer_size,
  "Pruning didn't make the hidden layer bigger");
print "New hidden layer size:  $pruned_hidden_layer_size\n";

# Test the test() method after growing and pruning:
$report = $nn->test("sample_data_nn.csv");

ok(length $report, "Test completed correctly after growing and pruning");
print $report;
print "____________\n";

# Test the run() method:
ok($nn->run("sample_data_nn_run.csv"), "Ran OK after growing and pruning");

# Test the save() method:
ok($nn->save("serialized.dat"), "Saved OK");

# Test the open() method:
my $restored_nn = MachineLearning::NeuralNetwork->open("serialized.dat");

ok($restored_nn->get_success(), "Successful serialized object restoration");
is($restored_nn->get_message(), "", "Empty error message, as expected");
is($restored_nn->get_name(), "Sample Neural Network", "Correct name");
like($restored_nn->get_description(), qr/network\z/is, "Correct description");
isa_ok($restored_nn, "MachineLearning::NeuralNetwork");
can_ok($restored_nn, $_) for ("run", "train", "test", "grow", "prune", "save");
ok(scalar(@{ $restored_nn->{"_input_neurons"} }) == 5, "Correct input layer size");
ok(scalar(@{ $restored_nn->{"_hidden_neurons"} })
  == $pruned_hidden_layer_size, "Correct hidden layer size");
ok(scalar(@{ $restored_nn->{"_output_neuron"} }) == 1, "Correct output layer size");
is($restored_nn->{"_input_neurons"}->[0]->{"Name"}, "First Serve Accuracy",
  "Correct first input neuron name");
is($restored_nn->{"_output_neuron"}->[0]->{"Name"}, "Result",
  "Correct output neuron name");
