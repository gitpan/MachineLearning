#!/usr/bin/perl

# dt_test.t:  Test the MachineLearning::DecisionTrees module.
use strict;
use utf8;
use English;
use Test::More ("tests" => 25);
use MachineLearning::DecisionTrees;

# Test the constructor and the accessor methods:
my $dt = MachineLearning::DecisionTrees->new({
  "InputFieldNames"  => ["First Serve Percentage", "Return Speed",
                         "Errors Per Set", "Height", "Weight"],
  "OutputFieldNames" => ["Singles Result", "Doubles Result"],
  "InputDataTypes" => {
    "First Serve Percentage" => "number",
    "Return Speed" => "number",
    "Errors Per Set" => "number",
    "Height" => "number",
    "Weight" => "number"},
  "TrainingData" => "sample_data_dt.csv"});
ok($dt->get_success(), "Successful object creation");
is($dt->get_message(), "", "Empty error message, as expected");
isa_ok($dt, "MachineLearning::DecisionTrees");
can_ok($dt, $_) for ("save", "print_out", "test", "employ");
ok(scalar(keys %{ $dt->{"_trees"} }) == 2, "Correct number of trees");
is($dt->{"_input_field_names"}->[0], "First Serve Percentage",
  "Correct first input field name");
is($dt->{"_output_field_names"}->[0], "Singles Result",
  "Correct first output field name");

# Test the print_out() method:
my $singles_result_printout = $dt->print_out("Singles Result");
my $doubles_result_printout = $dt->print_out("Doubles Result");

if (open my $TXT_FILE, ">", "singles_result_printout.txt") {
    print $TXT_FILE $singles_result_printout;
    close $TXT_FILE;
    ok(1, "Printed out the singles result OK");
}
else {
    ok(0, "Printed out the singles result OK");
} # end if

if (open my $TXT_FILE, ">", "doubles_result_printout.txt") {
    print $TXT_FILE $doubles_result_printout;
    close $TXT_FILE;
    ok(1, "Printed out the doubles result OK");
}
else {
    ok(0, "Printed out the doubles result OK");
} # end if

# Test the test() method:
my $report = $dt->test({
  "TreeName" => "Singles Result",
  "ValidationData" => "sample_data_dt.csv",
  "NodeList" => [6, 15]});

ok(length $report, "Test completed correctly");

unless ($dt->get_success()) {
    print STDERR $dt->get_message() . "\n";
    exit(1);
} # end unless

print $report;
print "____________\n";

# Test the employ() method:
ok($dt->employ({
  "TreeName" => "Singles Result",
  "TargetData" => "sample_data_dt_employ.csv",
  "NodeList" => [6, 15]}), "Employed OK");

unless ($dt->get_success()) {
    print STDERR $dt->get_message() . "\n";
    exit(1);
} # end unless

# Test the save() method:
ok($dt->save("serialized.dat"), "Saved OK");

unless ($dt->get_success()) {
    print STDERR $dt->get_message() . "\n";
    exit(1);
} # end unless

# Test the open() method:
my $restored_dt = MachineLearning::DecisionTrees->open("serialized.dat");

unless ($dt->get_success()) {
    print STDERR $dt->get_message() . "\n";
    exit(1);
} # end unless

ok($restored_dt->get_success(), "Successful serialized object restoration");
is($restored_dt->get_message(), "", "Empty error message, as expected");
isa_ok($restored_dt, "MachineLearning::DecisionTrees");
can_ok($restored_dt, $_) for ("save", "print_out", "test", "employ");
is($restored_dt->{"_input_field_names"}->[0], "First Serve Percentage",
  "Correct first input field name");
is($restored_dt->{"_output_field_names"}->[0], "Singles Result",
  "Correct first output field name");
is($restored_dt->{"_output_field_names"}->[1], "Doubles Result",
  "Correct second output field name");
