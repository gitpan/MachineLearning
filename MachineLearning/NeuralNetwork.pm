####
# NeuralNetwork.pm:  A Perl module defining a package for creating a
# MachineLearning::NeuralNetwork object.  This module uses the
# MachineLearning::SimulatedAnnealing module to train the neural network.
#
# Copyright 2009 by Benjamin Fitch
#
# This library is free software; you can redistribute it and/or modify it
# under the same terms as Perl itself.
####
package MachineLearning::NeuralNetwork;

use 5.008;
use strict;
use warnings;
use utf8;

use base ("MachineLearning");

use English "-no_match_vars";
use Hash::Util ("lock_keys");
use List::Util ("sum");
use Scalar::Util ("looks_like_number");
use Storable;

use MachineLearning::SimulatedAnnealing;

# Version:
our $VERSION = '1.03';

# Constants:
my $POUND     = "#";
my $SQ        = "'";
my $DQ        = "\"";
my $SEMICOLON = ";";

# Constructor:
sub new {
    my ($class, $args) = @_;
    my $nn;

    # Construct the object using the superclass constructor:
    $nn = $class->SUPER::new($args);

    # Restrict the keys in %{ $nn }:
    lock_keys(%{ $nn },
      "_success", "_message", "_name", "_description",
      "_input_field_names", "_output_field_names", "_is_trained",
      "_input_neurons", "_hidden_neurons", "_output_neuron");

    # Make sure that there is only one output field name:
    if (scalar(@{ $nn->{"_output_field_names"} }) > 1) {
        $nn->{"_success"} = 0;
        $nn->{"_message"} = "More than one output field name was passed "
          . "to the $class constructor.";
    } # end if

    # Initialize the private fields not handled by the superclass
    # constructor:
    $nn->{"_name"} = "";
    $nn->{"_description"} = "";
    $nn->{"_is_trained"} = 0;

    $nn->{"_input_neurons"}  = []; # hash refs with Name and Weights fields,
                                   # where Weights is a reference to an
                                   # array that is the same size as the
                                   # hidden layer

    $nn->{"_hidden_neurons"} = []; # hash refs with Threshold and Weights
                                   # fields, where Weights is a reference to
                                   # an array with a single element holding
                                   # the weight value to be fed into the
                                   # single-neuron output layer

    $nn->{"_output_neuron"} = [];  # holds a single hash reference with Name
                                   # and Threshold fields

    # Add the neuron names to the input and output layers:
    my %neuron_names = ();
    my $validate_neuron_name
      = sub {
        my $neuron_name = $_[0];

        if (exists $neuron_names{$neuron_name}) {
            $nn->{"_success"} = 0;
            $nn->{"_message"} = "All neuron names must be unique within a "
            . "neural network.";
            return 0;
        } # end if

        $neuron_names{$neuron_name} = 1;

        if ($neuron_name =~ /[,\r\n]/s) {
            $nn->{"_success"} = 0;
            $nn->{"_message"} = "Neuron names must not contain commas or "
              . "line-break characters.";
            return 0;
        } # end unless

        return 1;
      };

    for my $neuron_name (@{ $nn->{"_input_field_names"} }) {
        if ($validate_neuron_name->($neuron_name)) {
            push @{ $nn->{"_input_neurons"} }, {};
            $nn->{"_input_neurons"}->[-1]->{"Name"} = $neuron_name;
        }
        else {
            $nn->{"_success"} = 0;
            $nn->{"_message"} = "$DQ$neuron_name$DQ is not a valid "
              . "neuron name.";
            last;
        } # end if
    } # next $neuron_name

    for my $neuron_name (@{ $nn->{"_output_field_names"} }) {
        if ($validate_neuron_name->($neuron_name)) {
            push @{ $nn->{"_output_neuron"} }, {};
            $nn->{"_output_neuron"}->[-1]->{"Name"} = $neuron_name;
        }
        else {
            $nn->{"_success"} = 0;
            $nn->{"_message"} = "$DQ$neuron_name$DQ is not a valid "
              . "neuron name.";
            last;
        } # end if
    } # end for

    # Parse the constructor arguments not handled by the superclass
    # constructor, and update the "_name", "_description", and
    # "_hidden_neurons" fields:
    if (ref($args) eq "HASH"
      && exists($args->{"Name"})
      && length($args->{"Name"})
      && exists($args->{"Description"})
      && length($args->{"Description"})
      && exists($args->{"HiddenLayerSize"})
      && looks_like_number($args->{"HiddenLayerSize"})
      && int($args->{"HiddenLayerSize"}) >= 1) {
        $nn->{"_name"} = $args->{"Name"};
        $nn->{"_description"} = $args->{"Description"};

        for (1..int($args->{"HiddenLayerSize"})) {
            push @{ $nn->{"_hidden_neurons"} }, {};
        } # next integer
    }
    else {
        $nn->{"_success"} = 0;
        $nn->{"_message"} = "Invalid argument passed to the "
          . "$class constructor.";
    } # end if

    # Return the neural network object:
    return $nn;
} # end constructor

# Public methods:

sub get_name {
    return $_[0]->{"_name"};
} # end sub

sub get_description {
    return $_[0]->{"_description"};
} # end sub

sub run {
    my $self = $_[0];
    my $data_file_path = $_[1];
    my $data = $self->_parse_data_file($data_file_path, 1);
    my $data_with_no_field_names = $self->_convert_to_array_refs($data);

    unless ($self->get_success()) {
        return 0;
    } # end unless

    unless ($self->{"_is_trained"}) {
        $self->{"_success"} = 0;
        $self->{"_message"} = "An attempt has been made to run the "
          . "neural network even though it has not been trained.  As a "
          . "result, this neural network object is no longer viable.";
        return 0;
    } # end unless

    unless (scalar @{ $data }) {
        return 0;
    } # end unless

    # Calculate the outputs:
    for my $record (@{ $data_with_no_field_names }) {
        my $input_layer_size = scalar(@{ $self->{"_input_neurons"} });
        my $output_layer_size = scalar(@{ $self->{"_output_neuron"} });
        my @input_values = @{ $record }[0..($input_layer_size - 1)];
        my $output_value = $self->_calculate_output(\@input_values);

        for my $input_value (@input_values) {
            unless (looks_like_number($input_value)
              && $input_value >= -1 && $input_value <= 1) {
                $self->{"_success"} = 0;
                $self->{"_message"} = "At least one input value in "
                  . "$DQ$data_file_path$DQ is not a number in the range "
                  . "-1 to 1 inclusive.";
                return 0;
            } # end unless
        } # next $input_value

        splice @{ $record }, $input_layer_size, $output_layer_size,
          ($output_value);
    } # next $record

    # Overwrite the CSV file, and then return a true value if successful:
    my $new_csv_data = "";

    for my $input_neuron_index (0..$#{ $self->{"_input_neurons"} }) {
        unless ($input_neuron_index == 0) {
            $new_csv_data .= ",";
        } # end unless

        $new_csv_data
          .= $self->{"_input_neurons"}->[$input_neuron_index]->{"Name"};
    } # next $input_neuron_index

    for my $output_neuron (@{ $self->{"_output_neuron"} }) {
        $new_csv_data .= "," . $output_neuron->{"Name"};
    } # end for

    $new_csv_data .= "\n";

    for my $record (@{ $data_with_no_field_names }) {
        for my $dex (0..$#{ $record }) {
            unless ($dex == 0) {
                $new_csv_data .= ",";
            } # end unless

            $new_csv_data .= $record->[$dex];
        } # next $dex

        $new_csv_data .= "\n";
    } # next $record

    if (open my $CSV_FILE, ">", $data_file_path) {
        print $CSV_FILE $new_csv_data;
        close $CSV_FILE;
    }
    else {
        $self->{"_success"} = 0;
        $self->{"_message"} = "Couldn't open $DQ$data_file_path$DQ "
          . "for writing:\n\n    $!";
        return 0;
    } # end if

    return 1;
} # end sub

sub train {
    my $self = $_[0];
    my $args = $_[1];
    my $cycles_per_temperature;
    my $minimum_ones_percentage;
    my $data_file_path;
    my $data;
    my $ranges;
    my $annealing_results;

    my $cost_calculator
      = sub {
        my $input = $_[0];

        $self->_update_weights_and_thresholds($input);

        return $self->_calculate_cost($data, $minimum_ones_percentage);
      };

    unless ($self->get_success()) {
        return 0;
    } # end unless

    # Parse the named arguments:
    if (ref($args) eq "HASH"
      && exists($args->{"TrainingDataFilePath"})
      && exists($args->{"CyclesPerTemperature"})
      && looks_like_number($args->{"CyclesPerTemperature"})
      && int($args->{"CyclesPerTemperature"}) > 0) {
        $cycles_per_temperature = int $args->{"CyclesPerTemperature"};
        $data_file_path = $args->{"TrainingDataFilePath"};
        $data = $self->_parse_data_file($data_file_path, 0);

        unless (scalar @{ $data }) {
            return 0;
        } # end unless

        for my $record (@{ $data }) {
            for my $input_field_name (@{ $self->{"_input_field_names"} }) {
                my $input_value = $record->{$input_field_name};

                unless (looks_like_number($input_value)
                  && $input_value >= -1 && $input_value <= 1) {
                    $self->{"_success"} = 0;
                    $self->{"_message"} = "At least one input value in "
                      . "$DQ$data_file_path$DQ is not a number in the "
                      . "range -1 to 1 inclusive.";
                    return 0;
                } # end unless
            } # next $input_field_name
        } # next $record

        if (exists($args->{"MinimumOnesPercentage"})
          && looks_like_number($args->{"MinimumOnesPercentage"})
          && $args->{"MinimumOnesPercentage"} > 0
          && $args->{"MinimumOnesPercentage"} < 100) {
            $minimum_ones_percentage = $args->{"MinimumOnesPercentage"};
        }
        else {
            $minimum_ones_percentage = 0;
        } # end if
    }
    else {
        $self->{"_success"} = 0;
        $self->{"_message"} = "Invalid argument passed to the "
          . "train() method.";
        return 0;
    } # end if

    # Populate $ranges:
    $ranges = [];

    for my $input_neuron (@{ $self->{"_input_neurons"} }) {
        for (0..$#{ $self->{"_hidden_neurons"} }) {
            push @{ $ranges }, [-1, 1];
        } # next range
    } # next $input_neuron

    for my $hidden_neuron (@{ $self->{"_hidden_neurons"} }) {
        push @{ $ranges }, [-1, 1];
        push @{ $ranges },
          [0.00000001, scalar(@{ $self->{"_input_neurons"} })];
    } # next $hidden_neuron

    push @{ $ranges },
      [0.00000001, scalar(@{ $self->{"_hidden_neurons"} })];

    # Perform the annealing:
    $annealing_results = anneal({
      "Ranges" => $ranges,
      "CostCalculator" => $cost_calculator,
      "CyclesPerTemperature" => $cycles_per_temperature});

    # Update the neural network object's state using the annealing results:
    $self->_update_weights_and_thresholds($annealing_results);
    $self->{"_is_trained"} = 1;

    # Return a true value:
    return 1;
} # end sub

sub test {
    my $self = $_[0];
    my $data_file_path = $_[1];
    my $data = $self->_parse_data_file($data_file_path, 0);
    my $report = "";
    my $cost;

    unless ($self->get_success()) {
        return "";
    } # end unless

    unless ($self->{"_is_trained"}) {
        $self->{"_success"} = 0;
        $self->{"_message"} = "An attempt has been made to test the "
          . "neural network even though it has not been trained.  As a "
          . "result, this neural network object is no longer viable.";
        return "";
    } # end unless

    unless (scalar @{ $data }) {
        return 0;
    } # end unless

    # Validate the input values in the CSV data:
    for my $record (@{ $data }) {
        for my $input_field_name (@{ $self->{"_input_field_names"} }) {
            my $input_value = $record->{$input_field_name};

            unless (looks_like_number($input_value)
              && $input_value >= -1 && $input_value <= 1) {
                $self->{"_success"} = 0;
                $self->{"_message"} = "At least one input value in "
                  . "$DQ$data_file_path$DQ is not a number in the "
                  . "range -1 to 1 inclusive.";
                return 0;
            } # end unless
        } # next $input_field_name
    } # next $record

    # Calculate the cost:
    $cost = $self->_calculate_cost($data, 0);

    # Create the report and return it:
    $report .= "\nTest Results for " . $DQ . $self->get_name() . $DQ . "\n";
    $report .= "=" x (19 + length($self->get_name())) . "\n\n";

    for my $output_neuron (@{ $self->{"_output_neuron"} }) {
        $report .= "Cost for " . $DQ . $output_neuron->{"Name"} . $DQ
          . ":  " . $cost . "\n";
    } # end for

    return $report;
} # end sub

sub grow {
    my $self = $_[0];
    my $args = $_[1];
    my $candidate_sizes = [5, 8, 13, 21, 34, 55, 89, 144];
    my $new_hidden_layer_size;

    my $training_data_file_path;
    my $validation_data_file_path;
    my $cycles_per_temperature;
    my $minimum_ones_percentage;

    unless ($self->get_success()) {
        return 0;
    } # end unless

    # Parse the named arguments:
    if (ref($args) eq "HASH"
      && exists($args->{"TrainingDataFilePath"})
      && exists($args->{"ValidationDataFilePath"})
      && exists($args->{"CyclesPerTemperature"})
      && looks_like_number($args->{"CyclesPerTemperature"})
      && int($args->{"CyclesPerTemperature"}) > 0) {
        $cycles_per_temperature = int $args->{"CyclesPerTemperature"};
        $training_data_file_path = $args->{"TrainingDataFilePath"};
        $validation_data_file_path = $args->{"ValidationDataFilePath"};

        if (exists($args->{"MinimumOnesPercentage"})
          && looks_like_number($args->{"MinimumOnesPercentage"})
          && $args->{"MinimumOnesPercentage"} > 0
          && $args->{"MinimumOnesPercentage"} < 100) {
            $minimum_ones_percentage = $args->{"MinimumOnesPercentage"};
        }
        else {
            $minimum_ones_percentage = 0;
        } # end if
    }
    else {
        $self->{"_success"} = 0;
        $self->{"_message"} = "Invalid argument passed to the "
          . "grow() method.";
        return 0;
    } # end if

    # Pick the best size and return it:
    $new_hidden_layer_size = $self->_pick_the_best_size({
      "CandidateSizes" => $candidate_sizes,
      "TrainingDataFilePath" => $training_data_file_path,
      "ValidationDataFilePath" => $validation_data_file_path,
      "CyclesPerTemperature" => $cycles_per_temperature,
      "MinimumOnesPercentage" => $minimum_ones_percentage});

    return $new_hidden_layer_size;
} # end sub

sub prune {
    my $self = $_[0];
    my $args = $_[1];
    my $top = scalar(@{ $self->{"_hidden_neurons"} });
    my $bottom;
    my $candidate_sizes;
    my $new_hidden_layer_size;

    my $training_data_file_path;
    my $validation_data_file_path;
    my $cycles_per_temperature;
    my $minimum_ones_percentage;

    for (reverse 1..($top - 1)) {
        if ($_ =~ /\A(?:144|89|55|34|21|13|8|5|3|2|1)\z/s) {
            $bottom = $_;
            last;
        } # end if
    } # next number

    for my $candidate_size (reverse $bottom..$top) {
        push @{ $candidate_sizes }, $candidate_size;
    } # next $candidate_size

    unless ($self->get_success()) {
        return 0;
    } # end unless

    # Parse the named arguments:
    if (ref($args) eq "HASH"
      && exists($args->{"TrainingDataFilePath"})
      && exists($args->{"ValidationDataFilePath"})
      && exists($args->{"CyclesPerTemperature"})
      && looks_like_number($args->{"CyclesPerTemperature"})
      && int($args->{"CyclesPerTemperature"}) > 0) {
        $cycles_per_temperature = int $args->{"CyclesPerTemperature"};
        $training_data_file_path = $args->{"TrainingDataFilePath"};
        $validation_data_file_path = $args->{"ValidationDataFilePath"};

        if (exists($args->{"MinimumOnesPercentage"})
          && looks_like_number($args->{"MinimumOnesPercentage"})
          && $args->{"MinimumOnesPercentage"} > 0
          && $args->{"MinimumOnesPercentage"} < 100) {
            $minimum_ones_percentage = $args->{"MinimumOnesPercentage"};
        }
        else {
            $minimum_ones_percentage = 0;
        } # end if
    }
    else {
        $self->{"_success"} = 0;
        $self->{"_message"} = "Invalid argument passed to the "
          . "grow() method.";
        return 0;
    } # end if

    # Pick the best size and return it:
    $new_hidden_layer_size = $self->_pick_the_best_size({
      "CandidateSizes" => $candidate_sizes,
      "TrainingDataFilePath" => $training_data_file_path,
      "ValidationDataFilePath" => $validation_data_file_path,
      "CyclesPerTemperature" => $cycles_per_temperature,
      "MinimumOnesPercentage" => $minimum_ones_percentage});

    return $new_hidden_layer_size;
} # end sub

sub save {
    my $self = $_[0];
    my $file_path = $_[1];

    unless ($self->get_success()) {
        return 0;
    } # end unless

    unless ($self->{"_is_trained"}) {
        $self->{"_success"} = 0;
        $self->{"_message"} = "An attempt has been made to save the "
          . "neural network even though it has not been trained.  As a "
          . "result, this neural network object is no longer viable.";
        return 0;
    } # end unless

    unless (defined store($self, $file_path)) {
        $self->{"_success"} = 0;
        $self->{"_message"} = "Couldn't store data to $DQ$file_path$DQ.";
        return 0;
    }

    return 1;
} # end sub

sub open {
    my $class = $_[0];
    my $file_path = $_[1];
    my $retrieved_nn = retrieve($file_path);

    unless (defined $retrieved_nn) {
        $retrieved_nn = $class->new({});
        $retrieved_nn->{"_success"} = 0;
        $retrieved_nn->{"_message"} = "Could not restore the $class "
          . "object from $DQ$file_path$DQ.";
    }

    return $retrieved_nn;
} # end sub

# Private methods:

# In addition to the object reference passed in automatically, the private
# _calculate_cost() instance method takes a reference to an array of valid
# data records (hash references with fields matching the names of the input
# and output neurons) and, optionally, a positive number less than 100
# specifying the minimum ones percentage.
#
#   NOTE:  If there is no valid minimum ones percentage, the method uses a
#   value of 0.  However, if there are no "1" results at all for the
#   output neuron over the specified data set, the cost is automatically
#   100.  There must always be at least one occurrence of a "1" result for
#   the output neuron to avoid an automatic cost assessment of 100, even if
#   the minimum ones percentage is 0.
#
# The method calculates and returns the cost according to the neural
# network's current weights and thresholds.  If the network does not
# have associated weights and thresholds, the method returns undef.
sub _calculate_cost {
    my $self = $_[0];
    my $records = $_[1];
    my $minimum_ones_percentage = $_[2];
    my $cost;

    my @output_values;  # "1" or "0" values generated by the network

    my @output_results; # A result of 1 indicates a mismatch between the
                        # output value generated by the network and that
                        # provided by the training or validation data

    unless ($self->{"_is_trained"}) {
        return undef;
    } # end unless

    unless (defined($minimum_ones_percentage)
      && looks_like_number($minimum_ones_percentage)
      && $minimum_ones_percentage > 0
      && $minimum_ones_percentage < 100) {
        $minimum_ones_percentage = 0;
    } # end unless

    # Populate @output_results:
    for my $record (@{ $records }) {
        my @input_values;
        my $canned_output_value;
        my $generated_output_value;

        for my $input_neuron (@{ $self->{"_input_neurons"} }) {
            push @input_values, $record->{$input_neuron->{"Name"}};
        } # next $input_neuron

        for my $output_neuron (@{ $self->{"_output_neuron"} }) {
            $canned_output_value =  $record->{$output_neuron->{"Name"}};
        } # end for

        $generated_output_value = $self->_calculate_output(\@input_values);
        push @output_values, $generated_output_value;

        if ($generated_output_value == 0) {
            next;
        } # end if

        if ($generated_output_value == $canned_output_value) {
            push @output_results, 0;
        }
        else {
            push @output_results, 1;
        } # end if
    } # next $record

    # Calculate the cost:
    my $num_ones = sum(@output_values);
    my $ones_percentage = 100 * ($num_ones / scalar(@output_values));

    if ($num_ones == 0 || $ones_percentage < $minimum_ones_percentage) {
        $cost = 100;
    }
    else {
        $cost = 100 * (sum(@output_results) / scalar(@output_results));
    } # end if

    return $cost;
} # end sub

# In addition to the object reference passed in automatically, the private
# _calculate_output() instance method takes a reference to an array of
# valid input values in the correct order.  The method returns the
# corresponding output value calculated according to the neural network's
# current weights and thresholds.  If the network has not been trained,
# the method returns undef.
sub _calculate_output {
    my $self = $_[0];
    my @inputs = @{ $_[1] };
    my @hidden_layer; # cumulative input into the hidden neurons
    my @output_layer; # cumulative input into the output neuron
    my $output;

    unless ($self->{"_is_trained"}) {
        return undef;
    } # end unless

    # Populate @hidden_layer
    for my $input_index (0..$#inputs) {
        my @weights = @{ $self->{"_input_neurons"}->[
          $input_index]->{"Weights"} };

        for my $weight_index (0..$#weights) {
            $hidden_layer[$weight_index]
              += $inputs[$input_index] * $weights[$weight_index];
        } # next $weight_index
    } # next $input_index

    # Populate @output_layer:
    $output_layer[0] = 0;

    for my $hidden_neuron_index (0..$#hidden_layer) {
        if ($hidden_layer[$hidden_neuron_index] >= $self->{
          "_hidden_neurons"}->[$hidden_neuron_index]->{"Threshold"}) {
            my @weights = @{ $self->{"_hidden_neurons"}->[
              $hidden_neuron_index]->{"Weights"} };

            $output_layer[0] += $weights[0];
        } # end if
    } # next $hidden_neuron_index

    # Calculate and return the output:
    $output = 0;

    if ($output_layer[0] >= $self->{"_output_neuron"}->[0]->{"Threshold"}) {
        $output = 1;
    } # end if

    return $output;
} # end sub

# In addition to the object reference passed in automatically, the private
# _update_weights_and_thresholds() instance method takes a reference to a
# prevalidated array of weights and thresholds the values and quantities of
# which are legal for the neural network object on which this method was
# called.  The array must meet the following criteria:
#
#  *  The weight values for each input neuron are consecutive elements in
#     the array.
#
#  *  The weight values plus a threshold value for each hidden neuron are
#     consecutive elements in the array, with the weight values first.
#
#  *  The final element in the array is the threshold value for the
#     output neuron.
#
#  *  Within the array, the input neurons are represented in order, then the
#     hidden neurons are represented in order, and then the output neuron
#     is represented.
#
# The method updates the state of the neural network object on which the
# method was called.
sub _update_weights_and_thresholds {
    my $self = $_[0];
    my $input = $_[1];
    my $input_layer_size = scalar(@{ $self->{"_input_neurons"} });
    my $hidden_layer_size = scalar(@{ $self->{"_hidden_neurons"} });
    my $output_layer_size = scalar(@{ $self->{"_output_neuron"} });
    my $offset = 0;

    for my $input_neuron (@{ $self->{"_input_neurons"} }) {
        $input_neuron->{"Weights"} = [];

        for my $input_index ($offset..($offset + $hidden_layer_size - 1)) {
            push @{ $input_neuron->{"Weights"} }, $input->[$input_index];
        } # next $input_index

        $offset += $hidden_layer_size;
    } # next $input_neuron

    for my $hidden_neuron (@{ $self->{"_hidden_neurons"} }) {
        $hidden_neuron->{"Weights"} = [];

        for my $input_index ($offset..($offset + $output_layer_size - 1)) {
            push @{ $hidden_neuron->{"Weights"} }, $input->[$input_index];
        } # next $input_index

        $offset += $output_layer_size;
        $hidden_neuron->{"Threshold"} = $input->[$offset];
        $offset++;
    } #next $hidden_neuron

    for my $output_neuron (@{ $self->{"_output_neuron"} }) {
        $output_neuron->{"Threshold"} = $input->[$offset];
        $offset++;
    } # next $output_neuron

    $self->{"_is_trained"} = 1;
} # end sub

# In addition to the object reference passed in automatically, the private
# _pick_the_best_size() instance method takes a reference to a hash with the
# following fields:
#
#     CandidateSizes – Reference to an array of positive integers
#     representing hidden-layer sizes
#
#     TrainingDataFilePath – Path to a file containing valid training data
#
#     ValidationDataFilePath – Path to a file containing valid test data
#
#     CyclesPerTemperature – The number of randomization cycles performed at
#     each temperature level during the simulated annealing process (see the
#     MachineLearning::SimulatedAnnealing module)
#
#     MinimumOnesPercentage – The minimum percentage of "1" results that an
#     output neuron must produce over a data set to avoid an automatic cost
#     assessment of 100.  NOTE:  Regardless of this setting, a complete
#     absence of "1" results always results in a cost of 100.
#
# The method trains and tests the neural network using each hidden-layer
# size in turn while storing the cost for each based on the results returned
# by the test() method.  The method then chooses the size that produced the
# lowest cost (favoring the smaller of two sizes when there is a tie),
# retrains the network using that size, and returns the new size.
#
# If an error occurs, the method returns 0 after making sure that the
# _success and _message fields of the object on which the method was called
# have been updated.
sub _pick_the_best_size {
    my $self = $_[0];
    my $candidate_sizes = $_[1]->{"CandidateSizes"};
    my $training_data_file_path = $_[1]->{"TrainingDataFilePath"};
    my $validation_data_file_path = $_[1]->{"ValidationDataFilePath"};
    my $cycles_per_temperature = $_[1]->{"CyclesPerTemperature"};
    my $minimum_ones_percentage = $_[1]->{"MinimumOnesPercentage"};
    my $new_size = scalar(@{ $self->{"_hidden_neurons"} });
    my $winning_result = {"Size" => $new_size, "Cost" => 100};
    my @test_results;

    for my $candidate_size (@{ $candidate_sizes }) {
        my $report;
        my $cost;

        $self->_reset($candidate_size);
        $self->train({
          "TrainingDataFilePath" => $training_data_file_path,
          "CyclesPerTemperature" => $cycles_per_temperature,
          "MinimumOnesPercentage" => $minimum_ones_percentage});
        $report = $self->test($validation_data_file_path);

        unless ($report =~ /\bcost for [^\r\n]+:\s+(\S+)/is) {
            return 0;
        } # end unless

        $cost = $1;
        push @test_results,
          {"Size" => $candidate_size, "Cost" => $cost};
    } # next $candidate_size

    for my $test_result (@test_results) {
        if ($test_result->{"Cost"} < $winning_result->{"Cost"}
          || ($test_result->{"Cost"} == $winning_result->{"Cost"}
          && $test_result->{"Size"} < $winning_result->{"Size"})) {
            $winning_result = $test_result;
            $new_size = $test_result->{"Size"};
        } # end if
    } # next $test_result

    $self->_reset($new_size);
    $self->train({
      "TrainingDataFilePath" => $training_data_file_path,
      "CyclesPerTemperature" => $cycles_per_temperature,
      "MinimumOnesPercentage" => $minimum_ones_percentage});
    return $new_size;
} # end sub

# In addition to the object reference passed in automatically, the private
# _reset() instance method takes a hidden-layer size.  The method
# reconstructs the neural network object's hidden layer using the specified
# size, deletes all fields containing weights or thresholds, and sets the
# _is_trained field to false.
sub _reset {
    my $self = $_[0];
    my $new_hidden_layer_size = $_[1];

    $self->{"_hidden_neurons"} = [];

    for (1..$new_hidden_layer_size) {
        push @{ $self->{"_hidden_neurons"} }, {};
    } # next neuron

    for my $neuron (@{ $self->{"_input_neurons"} },
      @{ $self->{"_output_neuron"} }) {
        delete $neuron->{"Weights"};
        delete $neuron->{"Threshold"};
    } # next $neuron

    $self->{"_is_trained"} = 0;
} # end sub

# In addition to the object reference passed in automatically, the private
# _convert_to_array_refs() instance method takes a reference to an array of
# hash references storing input and output neuron values keyed by the neuron
# names.  The method returns a reference to an array of references to arrays
# containing the input neuron values in the correct order followed by the
# output neuron values in the correct order, with no field-name information.
sub _convert_to_array_refs {
    my $self = $_[0];
    my $hash_refs = $_[1];
    my $array_refs = [];

    for my $hash_ref (@{ $hash_refs }) {
        push @{ $array_refs }, [];

        for my $neuron (@{ $self->{"_input_neurons"} },
          @{ $self->{"_output_neuron"} }) {
            push @{ $array_refs->[-1] }, $hash_ref->{$neuron->{"Name"}};
        } # next $neuron
    } # next $hash_ref

    return $array_refs;
} # end sub

# Module return value:
1;
__END__

=head1 NAME

MachineLearning::NeuralNetwork - create, train, test, resize, store, and run a neural network

=head1 SYNOPSIS

  use MachineLearning::NeuralNetwork;
  my $nn = MachineLearning::NeuralNetwork->new({
    "Name" => "Sample Network",
    "Description" => "Sample network",
    "HiddenLayerSize" => 5,
    "InputFieldNames" => ["Input1", "Input2"],
    "OutputFieldNames" => ["Output"]});
  my $nn = MachineLearning::NeuralNetwork->open(
    "~/neural_networks/sample.nn");
  my $success = $nn->get_success();
  my $message = $nn->get_message();
  my $network_name = $nn->get_name();
  my $network_description = $nn->get_description();
  my $ran_ok = $nn->run($data_file_path);
  my $trained_ok = $nn->train({
    "TrainingDataFilePath" => $training_data_file_path,
    "CyclesPerTemperature" => $cycles_per_temperature,
    "MinimumOnesPercentage" => $minimum_ones_percentage});
  my $test_results = $nn->test($validation_data_file_path);
  my $new_hidden_layer_size = $nn->grow({
    "TrainingDataFilePath" => $training_data_file_path,
    "ValidationDataFilePath" => $validation_data_file_path,
    "CyclesPerTemperature" => $cycles_per_temperature,
    "MinimumOnesPercentage" => $minimum_ones_percentage});
  my $new_hidden_layer_size = $nn->prune({
    "TrainingDataFilePath" => $training_data_file_path,
    "ValidationDataFilePath" => $validation_data_file_path,
    "CyclesPerTemperature" => $cycles_per_temperature,
    "MinimumOnesPercentage" => $minimum_ones_percentage});
  my $saved_ok = $nn->save("~/neural_networks/perfected_network.nn");

=head1 DESCRIPTION

This module defines a package for creating a MachineLearning::NeuralNetwork
object.

This module uses the MachineLearning::SimulatedAnnealing module to optimize
the network's weights and thresholds during training.

A neural network as implemented by the MachineLearning::NeuralNetwork module
favors quality over quantity.  That is, it is optimized to find the highest
quality predictions for a single result value without particular regard to
how many data records (or instances) get screened out in the process.  This
is highly useful for many applications.  For example, out of many potential
financial investments, you might want to identify a small group that have
unusually good prospects for success.

The result values supported by the MachineLearning::NeuralNetwork module
are C<1> and C<0>, and the accuracy of the results is important only for the
C<1> values.  To ensure that the neural network's output layer generates a
satisfactory minimum quantity of ones, the methods for training, growing,
and pruning the neural network take an optional MinimumOnesPercentage
argument.

=head1 PREREQUISITES

To use this module, you must have both the MachineLearning module and the
MachineLearning::SimulatedAnnealing module installed.

=head1 METHODS

=over

=item MachineLearning::NeuralNetwork->new($args);

This is the constructor.

In addition to the class-name argument, which is passed automatically when
you use the C<MachineLearning::NeuralNetwork-E<gt>new()> syntax, the
constructor takes a reference to a hash containing the following keys:

  Name
  Description
  HiddenLayerSize
  InputFieldNames
  OutputFieldNames

The Name and Description must be non-empty strings.  The HiddenLayerSize
must be a positive integer specifying the number of neurons in the neural
network's hidden layer.  The value associated with the InputFieldNames key
must be a reference to an array of input field names.  The value associated
with the OutputFieldNames key must be a reference to an array containing
exactly one output field name.

All field names (for input and output fields combined) must be unique.
Field names must not contain commas or line-break characters.  There must be
at least two input field names and exactly one output field name.

The constructor returns a reference to a MachineLearning::NeuralNetwork
object, which is implemented internally as a hash.  All functionality is
exposed through methods.

If the constructor receives a valid hash reference providing all required
information, the C<get_success()> instance method returns true (C<1>) and
the C<get_message()> instance method returns an empty string; otherwise,
C<get_success()> returns false (C<0>) and C<get_message()> returns a string
containing an error message.

=item $nn->get_success();

This returns true (C<1>) if the neural network object was initialized
successfully; otherwise, it returns false (C<0>).

=item $nn->get_message();

When C<get_success()> returns true (C<1>), this returns an empty string.
When C<get_success()> returns false (C<0>), C<get_message()> returns an
error message.

=item $nn->get_name();

Returns the name of the neural network, or an empty string if the
neural network was never successfully initialized.

=item $nn->get_description();

Returns the description of the neural network, or an empty string if the
neural network was never successfully initialized.

=item $nn->run($data_file_path);

This method runs the neural network on the specified data.

The specified data file must be in CSV format with a header row.  The header
row must contain the names of the input neurons in the correct order
followed by the name of the output neuron.  B<NOTE:>  There can be more than
one output field; however, the neural network uses I<and preserves> only the
one for which the column heading matches the output-field name associated
with the neural network.

Each record in the data must contain the correct number of input values as
well as a blank or replaceable output value in the appropriate output field.
B<IMPORTANT:>  The method removes all other output fields from the file.

The method adds the neural network generated output value to each record,
overwriting the output value that is already there, if any.  B<NOTE:>  Input
and output values must not contain commas or line-break characters.

If everything goes OK, the method returns true (C<1>); otherwise, the method
returns false (C<0>).  If the neural network was in a valid state previously
but something went wrong during execution of the C<run()> method, the method
sets the C<_success> field (returned by the C<get_success()> method) to
false (C<0>) and places an error message in the C<_message> field (returned
by the C<get_message()> method).

=item $nn->train($args);

C<$args> is a reference to a hash containing the following keys:

  TrainingDataFilePath
  CyclesPerTemperature
  MinimumOnesPercentage

This method trains the neural network using the specified data and the
specified number of cycles per temperature.  The value specifying the
minimum percentage of ones required for the output node to avoid an
automatic cost assessment of C<100> is optional and, if missing or invalid,
is assumed to be C<0>.  Even if the minimum ones percentage is C<0>,
however, there must always be at least one occurrence of a C<1> result for
the output neuron to avoid an automatic cost assessment of C<100>.

The training data must be in CSV format with a header row.  The header row
must contain the names of the input neurons in the correct order followed by
the name of the output neuron.  B<NOTE:>  There can be more than one output
field; however, the neural network uses only the one for which the column
heading matches the output field name associated with the neural network.

Each record in the data must contain the correct number of input values as
well as an output value in the appropriate output field.  The output value
supplied with the training data is typically the expected, ideal, or
real-life result for the supplied input values.  B<NOTE:>  Input and output
values must not contain commas or line-break characters.

CyclesPerTemperature is a positive integer (for example, C<1_000>)
specifying the number of randomization cycles performed at each
temperature level during the simulated annealing process.  (For more
information, see the MachineLearning::SimulatedAnnealing module).

During training, the network minimizes I<cost> using simulated annealing to
optimize the weights and thresholds.  I<Cost> is a number that represents
how much error results when a particular set of weights and thresholds is
applied to the training data.  The cost is the percentage of the time that
the C<1> values in the output generated by the network do not match the
corresponding values provided by the training data.

If specified, the MinimumOnesPercentage value must be a positive number
less than 100 that represents a percentage.  The cost calculated for the
output neuron's values over a data set will be set to C<100> automatically
if the percentage of ones is less than the specified minimum (or if there
are no ones at all).

If everything goes OK, the method returns true (C<1>); otherwise, the method
returns false (C<0>).  If the neural network was in a valid state previously
but something went wrong during execution of the C<train()> method, the
method sets the C<_success> field (returned by the C<get_success()> method)
to false (C<0>) and places an error message in the C<_message> field
(returned by the C<get_message()> method).

=item $nn->test($validation_data_file_path);

This method tests the network using the supplied validation data, which
must be in the same format as the training data.

The method returns a string containing the test results in the form of
a formatted report, which gives the I<cost> for the output neuron.

I<Cost> is a number that represents how much error results when the neural
network is applied to the test data.  The cost is the percentage of the time
that the C<1> values in the output generated by the network do not match the
corresponding values provided by the test data.

During training, the network minimizes cost using simulated annealing to
optimize the weights and thresholds.  During testing, however, there are no
adjustments to the weights and thresholds; the results are simply calculated
and reported.

B<TIP:>  Testing reveals how well the network generalizes to out-of-sample
data.  Therefore, make sure that the validation data does not overlap with
the training data.  To compare the test results with the results of applying
the network to the data on which it was trained, you can run a test using
the training data.  The cost is typically higher for the test data, so the
important question is whether that cost is sufficiently low for the network
to be useful.

If something goes wrong during execution of the method, the method
returns an empty string and, if the C<_success> field (returned by the
C<get_success()> method) is currently set to true (C<1>), sets that field
to false (C<0>) and places an error message in the C<_message> field
(returned by the C<get_message()> method).

=item $nn->grow($args);

C<$args> is a reference to a hash containing the following keys:

  TrainingDataFilePath
  ValidationDataFilePath
  CyclesPerTemperature
  MinimumOnesPercentage

The MinimumOnesPercentage key is optional.

This method grows the neural network by performing training and testing with
a progressively increasing number of hidden neurons.  The size of the hidden
layer starts at five and then progresses upward through the Fibonacci series
to 144 (that is, the sizes used are 5, 8, 13, 21, 34, 55, 89, and 144).
Once the neural network has been trained and tested with a hidden layer
size of 144, the method chooses the size with the best result (the lowest
cost) based on post-training validation, retrains the network with that
number of hidden neurons, and then returns that number.  B<NOTE:>  In the
case of a tie, the method favors the smaller number of hidden neurons.

If something goes wrong during execution of the method, the method returns
C<0> and, if the C<_success> field (returned by the C<get_success()> method)
is currently set to true (C<1>), sets that field to false (C<0>) and places
an error message in the C<_message> field (returned by the C<get_message()>
method).

=item $nn->prune($args);

C<$args> is a reference to a hash containing the following keys:

  TrainingDataFilePath
  ValidationDataFilePath
  CyclesPerTemperature
  MinimumOnesPercentage

The MinimumOnesPercentage key is optional.

This method prunes the neural network by performing training followed by
testing with a progressively decreasing number of hidden neurons.  The size
of the hidden layer decreases by one for each cycle of training and testing.
Once all sizes have been tried from the initial size down to the closest
lower number that is in the Fibonacci series, the method chooses the size
with the best result (the lowest cost), retrains the network with that
number of hidden neurons, and returns that number.  B<NOTE:>  In the case of
a tie, the method favors the smaller number of hidden neurons.

If something goes wrong during execution of the method, the method returns
C<0> and, if the C<_success> field (returned by the C<get_success()> method)
is currently set to true (C<1>), sets that field to false (C<0>) and places
an error message in the C<_message> field (returned by the C<get_message()>
method).

=item $nn->save($file_path);

This method saves (I<serializes>) the neural network object to a file.
A neural network must be already trained before you can save it.

If the serialization and file-writing operations succeed, this method
returns true (C<1>); otherwise, the method sets the C<_success> field
to false (C<0>), places an error message in the C<_message> field, and
returns false (C<0>).

=item MachineLearning::NeuralNetwork->open($file_path);

This method returns a new MachineLearning::NeuralNetwork object created by
restoring such an object from the specified file.  If the file-reading
and deserialization operations succeed, the resulting object's
C<get_success()> method returns true (C<1>) and the C<get_message()> method
returns an empty string.  Otherwise, the C<open()> method creates and
returns a new object that has a false value in the C<_success> field and an
error message in the C<_message> field.

=back

=head1 NEURAL NETWORK DATA

All input values must be decimal numbers in the range C<-1> to C<1>,
inclusive.

Internally, the neural network uses weight values in the range C<-1> to
C<1> (inclusive) and thresholds in the range C<0.00000001> to I<n>, where
I<n> is the number of neurons in the preceding layer.  Both the hidden and
output layers have thresholds, and the output value is determined by
whether the threshold for the output node is reached (C<1> if yes,
C<0> if no).

All output values provided by training data or validation data must be
either C<0> or C<1>.

=head1 NEURAL NETWORK ARCHITECTURE

This module uses a feed-forward neural network architecture with one hidden
layer.  The number of hidden nodes is variable, and the recommended approach
is to try various numbers in ascending order (for example, by using the
C<grow()> method).  Then, starting with the number that produced the best
results based on post-training validation, prune the neural network using
the C<prune()> method.

B<TIP:>  You can grow and then prune a neural network several times using
different data sets in order to gain more insight into the optimal size for
the hidden layer.  You can also switch the training and validation sets to
get twice as many train-and-test cycles from your data.  When using these
approaches, consider reserving sufficient data for a final test; data that
is not part of any of the data sets that you are using for training and
validation during the development phase of the neural network.  If the final
test is not satisfactory, you might have to reconsider the types of inputs
that you are using for the neural network, gather sufficient additional data
for a new final test, and then develop the neural network again using a
different input framework.

=head1 AUTHOR

Benjamin Fitch, <blernflerkl@yahoo.com>

=head1 COPYRIGHT AND LICENSE

Copyright 2009 by Benjamin Fitch

This library is free software; you can redistribute it and/or modify it
under the same terms as Perl itself.

=cut
