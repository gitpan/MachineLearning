#!/usr/bin/perl

# anneal_test.t:  Test the MachineLearning::SimulatedAnnealing module.
use strict;
use utf8;
use English;
use Test::More ("tests" => 5);
use MachineLearning::SimulatedAnnealing;

my $cost_function
  = sub {
    my ($a, $b, $c) = @{ $_[0] };
    my $x = 3;
    my $poly = ($a * $x**2) + ($b * $x) + $c;
    my $cost = abs(23 - $poly);

    return $cost;
  };

my @ranges = ([0, 10], [0, 10], [-10, 0]);

my $cycles_1 = 1_000;
my $cycles_2 =   100;
my $cycles_3 =     1;

my $result_1 = anneal({
  "Ranges" => \@ranges,
  "CostCalculator" => $cost_function,
  "CyclesPerTemperature" => $cycles_1});

my $result_2 = anneal({
  "Ranges" => \@ranges,
  "CostCalculator" => $cost_function,
  "CyclesPerTemperature" => $cycles_2});

my $result_3 = anneal({
  "Ranges" => \@ranges,
  "CostCalculator" => $cost_function,
  "CyclesPerTemperature" => $cycles_3});

ok($result_1->[0] >= 0 && $result_1->[0] <= 10,
  "First number is in the prescribed range");
ok($result_1->[1] >= 0 && $result_1->[1] <= 10,
  "Second number is in the prescribed range");
ok($result_1->[2] >= -10 && $result_1->[2] <= 0,
  "Third number is in the prescribed range");
cmp_ok($cost_function->($result_1), "<", $cost_function->($result_2),
  "First result is better than the second result");
cmp_ok($cost_function->($result_2), "<", $cost_function->($result_3),
  "Second result is better than the third result");
