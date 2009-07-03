####
# DecisionTrees.pm:  A Perl module defining a package for creating a
# MachineLearning::DecisionTrees object.
#
# Copyright 2009 by Benjamin Fitch
#
# This library is free software; you can redistribute it and/or modify it
# under the same terms as Perl itself.
####
package MachineLearning::DecisionTrees;

use 5.008;
use strict;
use warnings;
use utf8;

use base ("MachineLearning");

use English "-no_match_vars";
use Hash::Util ("lock_keys");
use List::Util ("first");
use Scalar::Util ("looks_like_number");
use Storable;

# Version:
our $VERSION = '1.02';

# Constants:
my $POUND     = "#";
my $SQ        = "'";
my $DQ        = "\"";
my $SEMICOLON = ";";

# Constructor:
sub new {
    my ($class, $args) = @_;
    my $grove;
    my $training_data_file_path;

    # Construct the object using the superclass constructor:
    $grove = $class->SUPER::new($args);

    # Restrict the keys in %{ $grove }:
    lock_keys(%{ $grove },
      "_success", "_message",
      "_input_data_types", "_input_field_names", "_output_field_names",
      "_training_data", "_standard_deviations", "_means", "_trees");

    # Get the input data types and the training-data file path from $args:
    if (ref($args) eq "HASH"
      && exists($args->{"InputDataTypes"})
      && ref($args->{"InputDataTypes"}) eq "HASH"
      && scalar(keys %{ $args->{"InputDataTypes"} })
      == scalar(@{ $grove->{"_input_field_names"} })
      && scalar(grep {
        exists $args->{"InputDataTypes"}->{$_};
      } @{ $grove->{"_input_field_names"} })
      == scalar(@{ $grove->{"_input_field_names"} })
      && exists($args->{"TrainingData"})
      && length($args->{"TrainingData"})
      && (-f $args->{"TrainingData"})) {
        $grove->{"_input_data_types"} = $args->{"InputDataTypes"};
        $training_data_file_path = $args->{"TrainingData"};
    }
    else {
        $grove->{"_success"} = 0;
        $grove->{"_message"} = "Invalid argument passed to the "
          . "$class constructor.";
        return $grove;
    } # end if

    # Parse the training data:
    $grove->{"_training_data"} = $grove->_parse_data_file(
      $training_data_file_path, 0);

    unless ($grove->get_success()) {
        return $grove;
    } # end unless

    # Validate the input field values, and get the standard deviation and
    # mean for each number and special_number field:
    my %number_arrays;
    my %negative_special_number_arrays;
    my %positive_special_number_arrays;

    for my $record (@{ $grove->{"_training_data"} }) {
        for my $field_name (keys %{ $record }) {
            my $field_value = $record->{$field_name};
            my $data_type;

            unless (exists $grove->{"_input_data_types"}->{$field_name}) {
                next;
            } # end unless

            $data_type = $grove->{"_input_data_types"}->{$field_name};

            if (ref($data_type) eq "ARRAY") {
                unless (defined first { $_ eq $field_value; }
                  @{ $data_type }) {
                    $grove->{"_success"} = 0;
                    $grove->{"_message"} = "The training data passed to "
                      . "the $class constructor contains one or more "
                      . "input fields for which the value does not match "
                      . "the specified data type.";
                    return $grove;
                } # end unless
            }
            else {
                unless (looks_like_number $field_value) {
                    $grove->{"_success"} = 0;
                    $grove->{"_message"} = "The input data type "
                      . "information passed to the $class constructor "
                      . "is not valid for the $DQ$field_name$DQ field.";
                    return $grove;
                } # end unless

                if ($data_type eq "number") {
                    push @{ $number_arrays{$field_name} }, $field_value;
                }
                elsif ($data_type eq "special_number" && $field_value < 0) {
                    push @{ $negative_special_number_arrays{$field_name} },
                      $field_value;
                }
                elsif ($data_type eq "special_number" && $field_value > 0) {
                    push @{ $positive_special_number_arrays{
                      $field_name} }, $field_value;
                }
                elsif ($data_type eq "special_number"
                  && $field_value == 0) {
                    # Do nothing:
                    1;
                }
                else {
                    $grove->{"_success"} = 0;
                    $grove->{"_message"} = "The training data passed to "
                      . "the $class constructor contains one or more "
                      . "input fields for which the value does not match "
                      . "the specified data type.";
                    return $grove;
                } # end if
            } # end if
        } # next $field_name
    } # next $record

    for my $input_name (keys %number_arrays) {
        my $array_ref = $number_arrays{$input_name};
        my $sd = 0;
        my $mean = 0;

        if (scalar @{ $array_ref }) {
            $sd = $grove->SUPER::get_pop_std_dev($array_ref);
            $mean = $grove->SUPER::get_mean($array_ref);
        } # end if

        $grove->{"_standard_deviations"}->{$input_name}->{"Number"} = $sd;
        $grove->{"_means"}->{$input_name}->{"Number"} = $mean;
    } # next $input_name

    for my $input_name (keys %negative_special_number_arrays) {
        my $array_ref = $negative_special_number_arrays{$input_name};
        my $sd = 0;
        my $mean = 0;

        if (scalar @{ $array_ref }) {
            $sd = $grove->SUPER::get_pop_std_dev($array_ref);
            $mean = $grove->SUPER::get_mean($array_ref);
        } # end if

        $grove->{"_standard_deviations"}->{$input_name}->{
          "NegativeSpecialNumber"} = $sd;
        $grove->{"_means"}->{$input_name}->{
          "NegativeSpecialNumber"} = $mean;
    } # next $input_name

    for my $input_name (keys %positive_special_number_arrays) {
        my $array_ref = $positive_special_number_arrays{$input_name};
        my $sd = 0;
        my $mean = 0;

        if (scalar @{ $array_ref }) {
            $sd = $grove->SUPER::get_pop_std_dev($array_ref);
            $mean = $grove->SUPER::get_mean($array_ref);
        } # end if

        $grove->{"_standard_deviations"}->{$input_name}->{
          "PositiveSpecialNumber"} = $sd;
        $grove->{"_means"}->{$input_name}->{
          "PositiveSpecialNumber"} = $mean;
    } # next $input_name

    # Populate the grove:
    for my $output_field_name (@{ $grove->{"_output_field_names"} }) {
        $grove->{"_trees"}->{$output_field_name} = Tree->new({
          "ContainingGrove" => $grove,
          "OutputFieldName" => $output_field_name});
    } # next $output_field_name

    # Return the grove:
    return $grove;
} # end constructor

# Public methods:

sub save {
    my $self = $_[0];
    my $file_path = $_[1];

    unless ($self->get_success()) {
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
    my $retrieved_grove = retrieve($file_path);

    unless (defined $retrieved_grove) {
        $retrieved_grove = $class->new({});
        $retrieved_grove->{"_success"} = 0;
        $retrieved_grove->{"_message"} = "Could not restore the $class "
          . "object from $DQ$file_path$DQ.";
    }

    return $retrieved_grove;
} # end sub

sub print_out {
    my $self = $_[0];
    my $tree_name = $_[1];

    unless ($self->get_success()) {
        return "";
    } # end unless

    return $self->{"_trees"}->{$tree_name}->print_out();
} # end sub

sub test {
    my $self = $_[0];
    my $args = $_[1];

    unless ($self->get_success()) {
        return "";
    } # end unless

    unless (ref($args) eq "HASH" && exists($args->{"TreeName"})
      && exists($args->{"ValidationData"}) && exists($args->{"NodeList"})
      && (-f $args->{"ValidationData"})
      && ref($args->{"NodeList"}) eq "ARRAY") {
        $self->{"_success"} = 0;
        $self->{"_message"} = "Invalid argument passed to the "
          . "test() method.";
        return "";
    } # end unless

    return $self->{"_trees"}->{$args->{"TreeName"}}->test({
      "ValidationData" => $args->{"ValidationData"},
      "NodeList" => $args->{"NodeList"}});
} # end sub

sub employ {
    my $self = $_[0];
    my $args = $_[1];

    unless ($self->get_success()) {
        return 0;
    } # end unless

    unless (ref($args) eq "HASH" && exists($args->{"TreeName"})
      && exists($args->{"TargetData"}) && exists($args->{"NodeList"})
      && (-f $args->{"TargetData"})
      && ref($args->{"NodeList"}) eq "ARRAY") {
        $self->{"_success"} = 0;
        $self->{"_message"} = "Invalid argument passed to the "
          . "employ() method.";
        return 0;
    } # end unless

    return $self->{"_trees"}->{$args->{"TreeName"}}->employ({
      "TargetData" => $args->{"TargetData"},
      "NodeList" => $args->{"NodeList"}});
} # end sub

# Classes:

####
# In addition to the class name passed in automatically when you use the
# Tree->new() syntax, the Tree constructor takes a reference to a hash with
# the following fields:
#
#     ContainingGrove – A reference to the grove that contains the tree
#
#     OutputFieldName – The name of the output field on which the tree
#     is based
#
# A tree has a name, a reference to the grove that contains the tree, a
# root node (which in turn contains all the nodes in the tree), a collection
# of references to the leaf nodes, a node ID counter for generating node
# numbers, and a queue of priority input names.
#
#   NOTE:  The priority input names determine the first three input fields
#   that the tree uses to divide the input records among the children,
#   grandchildren, and great-grandchildren of the root node.
#
#   To determine the three priority input fields, the tree evaluates all
#   the possible permutations of input fields for the first three slots.
#   The "winning" permutation is the one resulting in the best winning node
#   among the children, grandchildren, and great-grandchildren of the root.
#   If there is a tie between permutations, the winner is the one resulting
#   in the best winning node among only the children and grandchildren of
#   the root node.  If there is a tie in that contest, the winner is the
#   permutation resulting in the best winning node among only direct
#   children of the root node.
#
#   "Best" is defined as having the hightest ones percentage or, where
#   the ones percentages are equal, having the highest quantity of records.
####
{
    package Tree;
    use strict;
    use utf8;
    use English "-no_match_vars";
    use Hash::Util ("lock_keys");

    sub new {
        my ($class, $args) = @_;
        my %tree;
        my $self;

        # Get the tree name and the containing grove from the arguments:
        $tree{"Name"} = $args->{"OutputFieldName"};
        $tree{"ContainingGrove"} = $args->{"ContainingGrove"};

        # Initialize the node ID counter:
        $tree{"NodeIdCounter"} = 0;

        # Bless the tree into the Tree class and get a reference:
        $self = bless \%tree, $class;

        # Restrict the keys in %tree:
        lock_keys(%{ $self }, "Name", "ContainingGrove", "RootNode",
          "LeafNodes", "NodeIdCounter", "PriorityInputs");

        # Build the tree:
        my $spawning_season;
        my %infertile_leaf_nodes;

        $self->{"RootNode"} = Node->new({"ContainingTree" => $self,
          "Records" => $self->{"ContainingGrove"}->{"_training_data"}});
        $self->{"LeafNodes"}->{$self->{"RootNode"}->{"IdNumber"}}
          = $self->{"RootNode"};
        $spawning_season = 1;

        while ($spawning_season) {
            my $spawning_has_occurred = 0;

            for my $node_id (keys %{ $self->{"LeafNodes"} }) {
                if (exists $infertile_leaf_nodes{$node_id}) {
                    next;
                }

                $self->{"LeafNodes"}->{$node_id}->spawn();

                if (scalar @{
                  $self->{"LeafNodes"}->{$node_id}->{"Children"} }) {
                    # Add the new children to the leaf node collection:
                    for my $newborn (@{
                      $self->{"LeafNodes"}->{$node_id}->{"Children"} }) {
                        $self->{"LeafNodes"}->{$newborn->{"IdNumber"}}
                          = $newborn;
                    } # next $newborn

                    # Remove the proud parent from the leaf node collection:
                    delete $self->{"LeafNodes"}->{$node_id};

                    # Let it be known that spawning has occurred:
                    $spawning_has_occurred = 1;
                }
                else {
                    $infertile_leaf_nodes{$node_id}
                      = $self->{"LeafNodes"}->{$node_id};
                } # end if
            } # next $node_id

            unless ($spawning_has_occurred) {
                $spawning_season = 0;
            } # end unless
        } # end while

        # Prune the tree:
        for my $leaf_node_id (keys %{ $self->{"LeafNodes"} }) {
            my $leaf_node;
            my @pruning_queue;

            unless (exists $self->{"LeafNodes"}->{$leaf_node_id}) {
                next;
            }

            $leaf_node = $self->{"LeafNodes"}->{$leaf_node_id};
            unshift @pruning_queue, $leaf_node;

            while (scalar @pruning_queue) {
                my $current_ones_percentage = 0;
                my @ancestor_evaluation_queue;

                unless (exists $pruning_queue[-1]->{"Parent"}) {
                    pop @pruning_queue;
                    next;
                } # end unless

                if ($pruning_queue[-1]->{"OnesQuantity"}) {
                    $current_ones_percentage
                      = 100 * ($pruning_queue[-1]->{"OnesQuantity"}
                      / ($pruning_queue[-1]->{"OnesQuantity"}
                      + $pruning_queue[-1]->{"ZeroesQuantity"}));
                } # end if

                unshift @ancestor_evaluation_queue,
                  $pruning_queue[-1]->{"Parent"};

                while (scalar @ancestor_evaluation_queue) {
                    my $ancestor_ones_percentage = 0;

                    if ($ancestor_evaluation_queue[-1]->{"OnesQuantity"}) {
                        $ancestor_ones_percentage
                          = 100 * ($ancestor_evaluation_queue[-1]->{
                          "OnesQuantity"}
                          / ($ancestor_evaluation_queue[-1]->{
                          "OnesQuantity"}
                          + $ancestor_evaluation_queue[-1]->{
                          "ZeroesQuantity"}));
                    } # end if

                    if ($ancestor_ones_percentage
                      >= $current_ones_percentage && exists(
                      $ancestor_evaluation_queue[-1]->{"Parent"})) {
                        for my $dex (reverse 0..$#{
                          $pruning_queue[-1]->{"Parent"}->{"Children"} }) {
                            if ($pruning_queue[-1]->{"Parent"}->{
                              "Children"}->[$dex]->{"IdNumber"}
                              == $pruning_queue[-1]->{"IdNumber"}) {
                                splice @{ $pruning_queue[-1]->{"Parent"}->{
                                  "Children"} }, $dex, 1;
                                last;
                            } # end if
                        } # next $dex

                        unless (scalar @{
                          $pruning_queue[-1]->{"Parent"}->{"Children"} }) {
                            my $parent = $pruning_queue[-1]->{"Parent"};

                            unshift @pruning_queue, $parent;
                            $self->{"LeafNodes"}->{$parent->{"IdNumber"}}
                              = $parent;
                        } # end unless

                        delete $self->{"LeafNodes"}->{
                          $pruning_queue[-1]->{"IdNumber"}};
                    }
                    elsif (exists(
                      $ancestor_evaluation_queue[-1]->{"Parent"})) {
                        unshift @ancestor_evaluation_queue,
                          $ancestor_evaluation_queue[-1]->{"Parent"};
                    } # end if

                    pop @ancestor_evaluation_queue;
                } # end while

                pop @pruning_queue;
            } # end while
        } # next $leaf_node_id

        # Get the priority inputs:
        my $highest_score = 0;
        my $current_quantity = 0;
        my $current_generation = 3;
        my @winning_queue;

        for my $input_one_candidate (@{
          $self->{"ContainingGrove"}->{"_input_field_names"} }) {
            my $level_zero_node = Node->new({"ContainingTree" => $self,
              "Records" => $self->{"ContainingGrove"}->{"_training_data"}});
            my $scores_quantities_and_records
              = $level_zero_node->get_scores_quantities_and_records(
                $input_one_candidate);
            my @generation_one_results;

            for my $input_value (keys %{
              $scores_quantities_and_records->{"Scores"} }) {
                push @generation_one_results, {};
                $generation_one_results[-1]->{"Score"}
                  = $scores_quantities_and_records->{"Scores"}->{
                  $input_value};
                $generation_one_results[-1]->{"Quantity"}
                  = $scores_quantities_and_records->{"Quantities"}->{
                    $input_value};
                $generation_one_results[-1]->{"Generation"} = 1;
            } # next $input_value

            $self->{"PriorityInputs"} = [$input_one_candidate,];
            $level_zero_node->spawn();

            for my $input_two_candidate (@{
              $self->{"ContainingGrove"}->{"_input_field_names"} }) {
                if ($input_two_candidate eq $input_one_candidate) {
                    next;
                } # end if

                # Get the results for this level, and spawn:
                my @generation_two_results;

                for my $level_one_node (@{
                  $level_zero_node->{"Children"} }) {
                    my $scores_quantities_and_records
                      = $level_one_node->get_scores_quantities_and_records(
                      $input_two_candidate);

                    for my $input_value (keys %{
                      $scores_quantities_and_records->{"Scores"} }) {
                        push @generation_two_results, {};
                      $generation_two_results[-1]->{"Score"}
                        = $scores_quantities_and_records->{"Scores"}->{
                        $input_value};
                      $generation_two_results[-1]->{"Quantity"}
                        = $scores_quantities_and_records->{"Quantities"}->{
                        $input_value};
                      $generation_two_results[-1]->{"Generation"} = 2;
                    } # next $input_value

                    if (scalar(@{
                      $self->{"ContainingGrove"}->{"_input_field_names"} })
                      > 2) {
                        $self->{"PriorityInputs"}
                          = [$input_one_candidate, $input_two_candidate];
                        $level_one_node->spawn();
                    } # end if
                } # next $level_one_node

                # If there are only two input fields, test this permutation
                # now and then skip the next level:
                for my $result (@generation_one_results,
                  @generation_two_results) {
                    if ($result->{"Score"} > $highest_score
                      || ($result->{"Score"} == $highest_score
                      && ($result->{"Quantity"} > $current_quantity
                      || ($result->{"Quantity"} == $current_quantity
                      && $result->{"Generation"}
                      < $current_generation)))) {
                        $highest_score = $result->{"Score"};
                        $current_quantity = $result->{"Quantity"};
                        $current_generation = $result->{"Generation"};
                        $winning_queue[0] = $input_one_candidate;
                        $winning_queue[1] = $input_two_candidate;
                    } # end if
                } # next $result

                next;

                # Go to the next level:
                for my $input_three_candidate (@{
                  $self->{"ContainingGrove"}->{"_input_field_names"} }) {
                    if ($input_three_candidate eq $input_one_candidate
                      || $input_three_candidate eq $input_two_candidate) {
                        next;
                    } # end if

                    # Get the results for this level:
                    my @generation_three_results;

                    for my $level_one_node (@{ $level_zero_node->{
                      "Children"} }) {
                        for my $level_two_node (@{ $level_one_node->{
                          "Children"} }) {
                            my $scores_quantities_and_records
                              = $level_two_node
                              ->get_scores_quantities_and_records(
                              $input_three_candidate);

                            for my $input_value (keys %{
                              $scores_quantities_and_records->{
                              "Scores"} }) {
                                push @generation_three_results, {};
                              $generation_three_results[-1]->{"Score"}
                                = $scores_quantities_and_records->{
                                "Scores"}->{$input_value};
                              $generation_three_results[-1]->{"Quantity"}
                                = $scores_quantities_and_records->{
                                "Quantities"}->{$input_value};
                              $generation_three_results[-1]->{"Generation"}
                                = 3;
                            } # next $input_value
                        } # next $level_two_node
                    } # next $level_one_node

                    # Test this permutation:
                    for my $result (@generation_one_results,
                      @generation_two_results, @generation_three_results) {
                        if ($result->{"Score"} > $highest_score
                          || ($result->{"Score"} == $highest_score
                          && ($result->{"Quantity"} > $current_quantity
                          || ($result->{"Quantity"} == $current_quantity
                          && $result->{"Generation"}
                          < $current_generation)))) {
                            $highest_score = $result->{"Score"};
                            $current_quantity = $result->{"Quantity"};
                            $current_generation = $result->{"Generation"};
                            $winning_queue[0] = $input_one_candidate;
                            $winning_queue[1] = $input_two_candidate;
                            $winning_queue[2] = $input_three_candidate;
                        } # end if
                    } # next $result
                } # next $input_three_candidate
            } # next $input_two_candidate
        } # next $input_one_candidate

        $self->{"PriorityInputs"} = \@winning_queue;

        # Return the tree:
        return $self;
    } # end constructor

    sub print_out {
        my $self = $_[0];
        my $printout = "";
        my @leaf_node_ids = keys %{ $self->{"LeafNodes"} };

        @leaf_node_ids
          = sort {
            my $a_quality = 0;
            my $b_quality = 0;
            my $a_quantity = scalar @{
              $self->{"LeafNodes"}->{$a}->{"Records"} };
            my $b_quantity = scalar @{
              $self->{"LeafNodes"}->{$b}->{"Records"} };
            my $result;

            if ($self->{"LeafNodes"}->{$a}->{"OnesQuantity"}) {
                $a_quality = 100 * ($self->{"LeafNodes"}->{$a}->{
                  "OnesQuantity"}
                  / ($self->{"LeafNodes"}->{$a}->{"OnesQuantity"}
                  + $self->{"LeafNodes"}->{$a}->{"ZeroesQuantity"}));
            } # end if

            if ($self->{"LeafNodes"}->{$b}->{"OnesQuantity"}) {
                $b_quality = 100 * ($self->{"LeafNodes"}->{$b}->{
                  "OnesQuantity"}
                  / ($self->{"LeafNodes"}->{$b}->{"OnesQuantity"}
                  + $self->{"LeafNodes"}->{$b}->{"ZeroesQuantity"}));
            } # end if

            $result
              = $b_quality > $a_quality ? 1
              : $b_quality < $a_quality ? -1
              : $b_quantity > $a_quantity ? 1
              : $b_quantity < $a_quantity ? -1
              : 0;
          } @leaf_node_ids;

        for my $leaf_node_id (@leaf_node_ids) {
            my $node_path
              = $self->{"LeafNodes"}->{$leaf_node_id}->get_path();
            my $records_quantity = scalar @{
              $self->{"LeafNodes"}->{$leaf_node_id}->{"Records"} };
            my $ones_percentage = 0;

            # Calculate the ones percentage:
            if ($self->{"LeafNodes"}->{$leaf_node_id}->{"OnesQuantity"}) {
                $ones_percentage = 100 * ($self->{"LeafNodes"}->{
                  $leaf_node_id}->{"OnesQuantity"}
                  / ($self->{"LeafNodes"}->{$leaf_node_id}->{"OnesQuantity"}
                  + $self->{"LeafNodes"}->{$leaf_node_id}->{
                  "ZeroesQuantity"}));
            } # end if

            # Format the ones percentage and the records quantity:
            $ones_percentage = sprintf "%.4f", $ones_percentage;
            $ones_percentage .= "\%";
            $records_quantity = ref($self)->_commacate_number(
              $records_quantity);

            # Add the information for this node to the printout string:
            $printout .= "\nNode $leaf_node_id\n";
            $printout .= "$ones_percentage\n";
            $printout .= "$records_quantity records\n";
            $printout .= "$node_path\n";
            $printout .= "____________\n";
        } # next $leaf_node_id

        $printout =~ s/_+\s+\z//s;

        return $printout;
    } # end sub

    sub test {
        my $self = $_[0];
        my $args = $_[1];
        my $validation_data = $args->{"ValidationData"};
        my $node_list = $args->{"NodeList"};
        my $records = $self->{"ContainingGrove"}->_parse_data_file(
          $validation_data, 0);
        my $output_field_name = $self->{"Name"};
        my $results = "";

        my $node_map; # hash reference; keys = node IDs; values = references
                      # to arrays of records; records = hash references

        unless ($self->{"ContainingGrove"}->get_success()) {
            return "";
        } # end unless

        $node_map = $self->_map_nodes_to_records($records);

        # Create a report comprising three key/value pairs, one per line,
        # where the keys and values are separated by equal signs and the
        # keys are 'Number of records', 'Number of "1" results', and
        # 'Accuracy of "1" results':
        my $records_count
          = ref($self)->_commacate_number(scalar @{ $records });
        my $one_results_count = 0;
        my $correct_results_count = 0;
        my $accuracy = 0;

        for my $node_id (@{ $node_list }) {
            $one_results_count += scalar @{ $node_map->{$node_id} };

            for my $record (@{ $node_map->{$node_id} }) {
                if ($record->{$output_field_name} == 1) {
                    $correct_results_count++;
                } # end if
            } # next $record
        } # next $node_id

        if ($one_results_count) {
            $accuracy = 100 * ($correct_results_count / $one_results_count);
            $accuracy = sprintf "%.4f", $accuracy;
            $one_results_count
              = ref($self)->_commacate_number($one_results_count);
        } # end if

        $results .= "Number of records=$records_count\n";
        $results .= "Number of ${DQ}1${DQ} results=$one_results_count\n";
        $results .= "Accuracy of ${DQ}1${DQ} results=$accuracy\%\n";

        # Return the report:
        return $results;
    } # end sub

    sub employ {
        my $self = $_[0];
        my $args = $_[1];
        my $target_data = $args->{"TargetData"};
        my $node_list = $args->{"NodeList"};
        my $records = $self->{"ContainingGrove"}->_parse_data_file(
          $target_data, 1);
        my $output_field_name = $self->{"Name"};
        my $new_csv_data = "";

        my $node_map; # hash reference; keys = node IDs; values = references
                      # to arrays of records; records = hash references

        unless ($self->{"ContainingGrove"}->get_success()) {
            return 0;
        } # end unless

        # Map the nodes to records:
        $node_map = $self->_map_nodes_to_records($records);

        # Prepare the records by setting the appropriate output field value
        # to "0" throughout the data set:
        for my $record (@{ $records }) {
            $record->{$output_field_name} = 0;
        } # next $record

        # Set the appropriate output field value to "1" in each record
        # associated with one of the user-specified nodes:
        for my $node_id (@{ $node_list }) {
            for my $record (@{ $node_map->{$node_id} }) {
                $record->{$output_field_name} = 1;
            } # next $record
        } # next $node_id

        # Create the new CSV data from the modified records:
        my @field_names
          = (@{ $self->{"ContainingGrove"}->{"_input_field_names"} },
          @{ $self->{"ContainingGrove"}->{"_output_field_names"} });

        for my $dex (0..$#field_names) {
            $new_csv_data .= $field_names[$dex];

            unless ($dex == $#field_names) {
                $new_csv_data .= ",";
            } # end unless
        } # next $dex

        $new_csv_data .= "\n";

        for my $record (@{ $records }) {
            for my $dex (0..$#field_names) {
                $new_csv_data .= $record->{$field_names[$dex]};

                unless ($dex == $#field_names) {
                    $new_csv_data .= ",";
                } # end unless
            } # next $dex

            $new_csv_data .= "\n";
        } # next $record

        # Overwrite the target file with the new CSV data:
        if (open my $CSV_FILE, ">", $target_data) {
            print $CSV_FILE $new_csv_data;
            close $CSV_FILE;
        }
        else {
            $self->{"ContainingGrove"}->{"_success"} = 0;
            $self->{"ContainingGrove"}->{"_message"} = "Couldn't open "
              . "$DQ$target_data$DQ for writing:\n\n    $!";
            return 0;
        } # end if

        # Return a "true" value indicating that everything went OK:
        return 1;
    } # end sub

    # Private methods:

    # In addition to the object reference passed in automatically, the
    # private _map_nodes_to_records() instance method takes a reference to
    # an array of records.  The method returns a node map.
    sub _map_nodes_to_records {
        my $self = $_[0];
        my $records = $_[1];
        my $node_map = {};
        my @queue = ();

        $node_map->{$self->{"RootNode"}->{"IdNumber"}} = $records;
        unshift @queue, $self->{"RootNode"};

        while (scalar(@queue) > 0) {
            if (scalar(@{ $queue[-1]->{"Children"} }) == 0) {
                pop @queue;
                next;
            } # end if

            # For each child of the current node, create an array of
            # records, add an entry to the node map, and unshift the child
            # onto the queue:
            for my $child (@{ $queue[-1]->{"Children"} }) {
                my $parent_records = $node_map->{$queue[-1]->{"IdNumber"}};
                my $child_records = [];
                my $input_name = $child->{"InputName"};
                my $input_value = $child->{"InputValue"};

                for my $parent_record (@{ $parent_records }) {
                    my $translated_field_value = $child->_get_input_value(
                      $input_name, $parent_record->{$input_name});

                    if ($translated_field_value eq $input_value) {
                        push @{ $child_records }, $parent_record;
                    } # end if
                } # next $parent_record

                $node_map->{$child->{"IdNumber"}} = $child_records;
                unshift @queue, $child;
            } # next $child

            # Remove the current node from the queue:
            pop @queue;
        } # end while

        return $node_map;
    } # end sub

    # In addition to the class name passed in automatically, the private
    # _commacate_number() class method takes a number string and returns
    # a string in which commas have been inserted for readability.
    sub _commacate_number {
        shift; # Discard the class name
        my $num_string = $_[0];

        $num_string = reverse $num_string;
        $num_string =~ s/(\d\d\d)(?=\d)(?!\d*\.)/$1,/gs;
        $num_string = reverse $num_string;

        return $num_string;
    } # end sub
} # end class

####
# In addition to the class name passed in automatically when you use the
# Node->new() syntax, the Node constructor takes a reference to a hash with
# the following fields:
#
#     ContainingTree – A reference to the tree that contains the node
#     Parent – A reference to the node's parent (omitted for a root node)
#     Records – A reference to an array of associated training-data records
#     InputName – The name of the associated input field (omitted for root)
#     InputValue – The associated input value (omitted for a root node)
#
# A node has a parent (unless the node is a root node), an array of children
# (empty if the node is a leaf node), an ID number, a ones quantity, a
# zeroes quantity, a set of records, an input name (unless the node is a
# root node), an input value (unless the node is a root node), and a list of
# input names available for descendants.
####
{
    package Node;
    use strict;
    use utf8;
    use English "-no_match_vars";
    use Hash::Util ("lock_keys");

    sub new {
        my ($class, $args) = @_;
        my %node;

        # Bless the node into the Node class and get a reference:
        my $self = bless \%node, $class;

        # Restrict the keys in %node:
        lock_keys(%{ $self }, "IdNumber", "ContainingTree", "Parent",
          "Children", "Records", "OnesQuantity", "ZeroesQuantity",
          "InputName", "InputValue", "AvailableInputNames");

        # Parse the arguments and initialize the fields:
        $self->{"ContainingTree"} = $args->{"ContainingTree"};
        $self->{"IdNumber"} = $self->{"ContainingTree"}->{"NodeIdCounter"};
        $self->{"ContainingTree"}->{"NodeIdCounter"}++;

        if (exists $args->{"Parent"}) {
            $self->{"Parent"} = $args->{"Parent"};
        } # end if

        $self->{"Children"} = [];
        $self->{"Records"} = $args->{"Records"};
        $self->{"OnesQuantity"} = 0;
        $self->{"ZeroesQuantity"} = 0;

        for my $record (@{ $self->{"Records"} }) {
            if ($record->{$self->{"ContainingTree"}->{"Name"}} == 1) {
                $self->{"OnesQuantity"}++;
            }
            else {
                $self->{"ZeroesQuantity"}++;
            } # end if
        } # next $record

        if (exists $args->{"InputName"}) {
            $self->{"InputName"} = $args->{"InputName"};
            $self->{"InputValue"} = $args->{"InputValue"};
        } # end if

        if (exists $self->{"Parent"}) {
            my @available_input_names = grep {
              $_ ne $self->{"InputName"};
            } @{ $self->{"Parent"}->{"AvailableInputNames"} };

            $self->{"AvailableInputNames"} = \@available_input_names;
        }
        else {
            $self->{"AvailableInputNames"} = $self->{"ContainingTree"}->{
              "ContainingGrove"}->{"_input_field_names"};
        } # end if

        # Return the node:
        return $self;
    } # end constructor

    sub spawn {
        my $self = $_[0];
        my $winning_input_name = "";

        my %input_scores;       # keys = input names; values = hashrefs:
                                #   keys = unique input values;
                                #   values = ones percentages

        my %input_quantities;   # keys = input names; values = hashrefs:
                                #   keys = unique input values;
                                #   values = record quantities

        my %input_records;      # keys = input names; values = hashrefs:
                                #   keys = unique input values;
                                #   values = array references

        my %winning_scores;     # keys = input names; values = scores

        my %winning_quantities; # keys = input names; values = quantities

        unless (scalar(@{ $self->{"Records"} }) >= 2) {
            return;
        } # end unless

        # Determine whether the node is the root node, a child of the root
        # node, or a grandchild of the root node, and if so, derive the
        # winning input name from the appropriate priority input (if it
        # exists) for the tree:
        my $generation = 0;
        my $current_node = $self;

        while (exists $current_node->{"Parent"}) {
            $current_node = $current_node->{"Parent"};
            $generation++;
        } # end while

        if (exists(
          $self->{"ContainingTree"}->{"PriorityInputs"}->[$generation])) {
            $winning_input_name = $self->{"ContainingTree"}->{
              "PriorityInputs"}->[$generation];
        } # end if

        # Get the scores (ones percentages), quantities (numbers of
        # records), and records for each available input name (or for
        # just the winning input name if it is known):
        for my $input_name (@{ $self->{"AvailableInputNames"} }) {
            my $scores_quantities_and_records;

            unless (length($winning_input_name) == 0
              || $input_name eq $winning_input_name) {
                next;
            } # end unless

            $scores_quantities_and_records
              = $self->get_scores_quantities_and_records($input_name);
            $input_scores{$input_name}
              = $scores_quantities_and_records->{"Scores"};
            $input_quantities{$input_name}
              = $scores_quantities_and_records->{"Quantities"};
            $input_records{$input_name}
              = $scores_quantities_and_records->{"Records"};
            $winning_scores{$input_name}
              = $scores_quantities_and_records->{"WinningScore"};
            $winning_quantities{$input_name}
              = $scores_quantities_and_records->{"WinningQuantity"};
        } # next $input_name

        # If the winning input name is yet to be determined, determine it:
        unless (length $winning_input_name) {
            # Choose the winning input, if any, or return without spawning:
            my $high_score = 0;
            my $current_quantity = 0;

            for my $input_name (@{ $self->{"AvailableInputNames"} }) {
                if ($winning_scores{$input_name} > $high_score
                  || ($winning_scores{$input_name} == $high_score
                  && $winning_quantities{$input_name}
                  > $current_quantity)) {
                    $winning_input_name = $input_name;
                    $high_score = $winning_scores{$input_name};
                    $current_quantity = $winning_quantities{$input_name};
                } # end if
            } # next $input_name

            unless ($high_score > 0) {
                return;
            } # end unless
        } # end unless

        # Spawn one or more children who have all their fingers (one or
        # more records) and toes (one or more "1" results):
        my $winning_scores = $input_scores{$winning_input_name};
        my $winning_quantities = $input_quantities{$winning_input_name};
        my $winning_records = $input_records{$winning_input_name};

        for my $input_value (keys %{ $winning_records }) {
            unless ($winning_scores->{$input_value} > 0
              && $winning_quantities->{$input_value} > 0) {
                next;
            } # end unless

            # Deliver the child:
            my $child = Node->new({
              "ContainingTree" => $self->{"ContainingTree"},
              "Parent" => $self,
              "Records" => $winning_records->{$input_value},
              "InputName" => $winning_input_name,
              "InputValue" => $input_value});

            push @{ $self->{"Children"} }, $child;
        } # next $input_value
    } # end sub

    sub get_scores_quantities_and_records {
        my $self = $_[0];
        my $input_name = $_[1];

        my $winning_score;
        my $winning_quantity;

        my %record_bin;
        my %ones_counts;
        my %scores_quantities_and_records;

        lock_keys(%scores_quantities_and_records,
          "Scores", "Quantities", "Records",
          "WinningScore", "WinningQuantity");

        for my $record (@{ $self->{"Records"} }) {
            my $field_value = $record->{$input_name};
            my $input_value
              = $self->_get_input_value($input_name, $field_value);

            push @{ $record_bin{$input_value} }, $record;

            if ($record->{$self->{"ContainingTree"}->{"Name"}} == 1) {
                $ones_counts{$input_value}++;
            } # end if
        } # next $record

        for my $input_value (keys %record_bin) {
            my $quantity = scalar @{ $record_bin{$input_value} };
            my $ones_count = $ones_counts{$input_value};
            my $ones_percentage;

            unless (defined $ones_count) {
                $ones_count = 0;
            } # end unless

            if ($quantity > 0) {
                $ones_percentage = 100 * ($ones_count / $quantity);
            }
            else {
                $ones_percentage = 0;
            } # end if

            $scores_quantities_and_records{"Scores"}->{$input_value}
              = $ones_percentage;
            $scores_quantities_and_records{"Quantities"}->{$input_value}
              = $quantity;
            $scores_quantities_and_records{"Records"}->{$input_value}
              = $record_bin{$input_value};

            $winning_score = 0;
            $winning_quantity = 0;

            if ($ones_percentage > $winning_score
              || ($ones_percentage == $winning_score
              && $quantity > $winning_quantity)) {
                $winning_score = $ones_percentage;
                $winning_quantity = $quantity;
            } # end if
        } # next $input_value

        $scores_quantities_and_records{"WinningScore"} = $winning_score;
        $scores_quantities_and_records{"WinningQuantity"}
          = $winning_quantity;

        return \%scores_quantities_and_records;
    } # end sub

    sub get_path {
        my $self = $_[0];
        my $path = "";
        my @way_stations;
        my @node_processing_queue;

        unshift @node_processing_queue, $self;

        while (scalar @node_processing_queue) {
            if (exists $node_processing_queue[-1]->{"Parent"}) {
                unshift @way_stations, $node_processing_queue[-1];
                unshift @node_processing_queue,
                  $node_processing_queue[-1]->{"Parent"};
            } # end if

            pop @node_processing_queue;
        } # end while

        $path .= "ROOT_NODE";

        for my $way_station (@way_stations) {
            $path .= " --> " . $way_station->{"InputName"} . "="
              . $way_station->{"InputValue"};
        } # next $way_station

        return $path;
    } # end sub

    # Private methods:

    # In addition to the object reference passed in automatically, the
    # private _get_input_value() instance method takes an input field name
    # and a field value and returns the corresponding enumeration-style
    # input value.
    sub _get_input_value {
        my $self = $_[0];
        my $field_name = $_[1];
        my $field_value = $_[2];

        my $grove = $self->{"ContainingTree"}->{"ContainingGrove"};
        my $field_type = $grove->{"_input_data_types"}->{$field_name};
        my $input_value = "";

        if (ref($field_type) eq "ARRAY") {
            $input_value = $field_value;
        }
        elsif ($field_type eq "number") {
            my $mean
              = $grove->{"_means"}->{$field_name}->{"Number"};
            my $dev
              = $grove->{"_standard_deviations"}->{$field_name}->{"Number"};

            $input_value
              = $field_value < $mean - (2 * $dev) ? "abnormally_low"
              : $field_value < $mean - $dev ? "low"
              : $field_value <= $mean + $dev ? "medium"
              : $field_value <= $mean + (2 * $dev) ? "high"
              : "abnormally_high";
        }
        elsif ($field_value < 0) {
            my $mean = $grove->{"_means"}->{$field_name}->{
              "NegativeSpecialNumber"};
            my $dev = $grove->{"_standard_deviations"}->{$field_name}->{
              "NegativeSpecialNumber"};

            $input_value
              = $field_value < $mean - (2 * $dev)
              ? "abnormally_strong_negative"
              : $field_value < $mean - $dev ? "strong_negative"
              : $field_value <= $mean + $dev ? "medium_negative"
              : $field_value <= $mean + (2 * $dev) ? "weak_negative"
              : "abnormally_weak_negative";
        }
        elsif ($field_value == 0) {
            $input_value = "abnormally_weak_positive_or_zero";
        }
        else {
            my $mean = $grove->{"_means"}->{$field_name}->{
              "PositiveSpecialNumber"};
            my $dev = $grove->{"_standard_deviations"}->{$field_name}->{
              "PositiveSpecialNumber"};

            $input_value
              = $field_value < $mean - (2 * $dev)
              ? "abnormally_weak_positive_or_zero"
              : $field_value < $mean - $dev ? "weak_positive"
              : $field_value <= $mean + $dev ? "medium_positive"
              : $field_value <= $mean + (2 * $dev) ? "strong_positive"
              : "abnormally_strong_positive";
        } # end if

        return $input_value;
    } # end sub
} # end class

# Module return value:
1;
__END__

=head1 NAME

MachineLearning::DecisionTrees - create, test, print out, store, and employ decision trees

=head1 SYNOPSIS

  use MachineLearning::DecisionTrees;
  my $grove = MachineLearning::DecisionTrees->new({
    "InputFieldNames" => ["Input1", "Input2", "Input3"],
    "OutputFieldNames" => ["Output1", "Output2"],
    "InputDataTypes" => {
      "Input1" => ["male", "female"],
      "Input2" => "number",
      "Input3" => "special_number"},
    "TrainingData" => $training_data_file_path});
  my $all_is_well = $grove->get_success();
  my $message = $grove->get_message();
  my $saved_ok = $grove->save($file_path);
  my $grove = MachineLearning::DecisionTrees->open($file_path);
  my $tree_printout = $grove->print_out($tree_name);
  my $tree_test_results = $grove->test({
    "TreeName" => $tree_name,
    "ValidationData" => $validation_data_file_path,
    "NodeList" => \@selected_node_numbers});
  my $completed_ok = $grove->employ({
    "TreeName" => $tree_name,
    "TargetData" => $writable_data_file_path,
    "NodeList" => \@selected_node_numbers});

=head1 DESCRIPTION

This module defines a package for creating a MachineLearning::DecisionTrees
object.

Decision trees as implemented by the MachineLearning::DecisionTrees module
favor quality over quantity.  That is, they are optimized to find the
highest quality predictions for a single output value without undue regard
to how many data records (or I<instances>) get screened out in the process.
This approach is highly useful for many applications.  For example, you
might want to narrow down a list of applicants for a grant while trying to
predict which recipients will make exceptional use of the grant; or, out of
many potential financial investments, you might want to identify a small
group that have unusually good prospects for success.

The output values supported by the MachineLearning::DecisionTrees module are
C<1> and C<0>, and the accuracy of the results is important only for the
C<1> values.  When choosing the attribute on which to split a node, the
criterion is maximization of the highest resulting ones percentage. (In the
case of a tie, the highest quantity of records breaks the tie.)
B<NOTE:>  The selection of the first three attributes (resulting in the
children, grandchildren, and great-grandchildren of the root node) is
performed organically.  That is, the best out of all possible permutations
of attributes for the first three slots "wins".  Subsequent attributes are
chosen individually.

Decision trees as implemented by the MachineLearning::DecisionTrees module
come in groves.  A grove is one or more decision trees, all built from the
same data file.  The data file used to build (or I<train>) a grove contains
one or more output fields, and each output field corresponds to one tree.
When printed out, a decision tree identifies itself using the name of the
output field for which it was built.

A decision tree is made up of nodes, starting with a root node (which
typically appears at the top in graphical representations, highlighting the
fact that trees in nature are upside down :-) and branching from there via
child nodes.  During the building, testing, or employment of a decision
tree, each parent node divides data records associated with it among its
children according to each record's value for a particular input field (or
I<attribute>).  A node with no children is a I<leaf node> and terminates a
particular path through the tree.

A I<path> through a decision tree always starts at the root node and ends
when one of the following conditions occurs during tree creation, which
comprises I<training> and I<pruning> phases:

=over

=item *

During training, there are no further attributes available with which to
generate children from the current node.

=item *

There is only one record associated with the current node.  B<NOTE:>  The
training algorithm never creates a child node that does not have any records
associated with it or that does not produce any "1" results.

=item *

The pruning process has determined that application of the remaining
attributes to the current branch does not improve its performance.

=back

=head2 Pruning

After a tree has been built, it gets I<pruned> from the bottom up.  The
pruning algorithm removes any leaf node if the records associated with one
of the node's ancestors (other than the root node) have an equal or higher
percentage of ones for the output field on which the tree is based than do
the records associated with the leaf node.  Any nodes that become leaf nodes
during the pruning process are themselves subject to pruning.

Once a tree has been pruned, the remaining leaf nodes are sorted by quality
(defined as the magnitude of the ones percentage for the records associated
with the leaf node).  In a printout of the tree, the leaf nodes appear in
order by quality (highest to lowest) together with the exact ones percentage
for each leaf node based on the training data.  For each leaf node, the
printout also gives the number of associated training records and the path
through the tree.

=head2 Input fields and output fields

The data used to build, test, or employ decision trees must include at least
two I<input fields>, which are used as attributes by nodes during spawning.

The data must also include one or more I<output fields> in which each value
is either a C<1> or a C<0>.  (The output values can be blank when the data
is passed to the C<employ()> method.)  For example, there might be a field
for whether you enjoy certain TV shows that have been on in the past and
another field for whether your spouse enjoys them.  You could create a grove
of two decision trees from this data:  one tree to select shows that you
would most likely enjoy out of the new television offerings starting up in
the fall, and another tree to select shows that your spouse would most
likely enjoy.  (With luck, there will be some that you I<both> enjoy!)

=head2 Input data types

Each input field uses one of three data types:  C<enumeration>, C<number>,
or C<special_number>.

An C<enumeration> type comprises a list of two or more possible unique text
values (for example, C<male> and C<female> for a Gender field).

A C<number> type comprises decimal numbers.  The tree-building algorithm
divides the values in a number field into the following categories after
statistically analyzing all the values for that field in the data set.
(In the B<Criteria> column below, C<m> is the mean and C<d> is the
population standard deviation.)

    Category           Criteria
    --------           --------

    abnormally_low     Less than m - 2d

    low                Greater than or equal to m - 2d, and less than
                       m - d

    medium             Greater than or equal to m - d, and less than
                       or equal to m + d (that is, within one standard
                       deviation of the mean)

    high               Greater than m + d, and less than or equal to
                       m + 2d

    abnormally_high    Greater than m + 2d

A C<special_number> type is a decimal number for which the sign and the
absolute value have separate significance.  For example, the sign might
indicate direction (up or down) while the absolute value indicates momentum,
or the sign might indicate approval versus disapproval while the absolute
value indicates the intensity of the response.  The categories for this type
are C<abnormally_strong_negative>, C<strong_negative>, C<medium_negative>,
C<weak_negative>, C<abnormally_weak_negative>,
C<abnormally_weak_positive_or_zero>, C<weak_positive>, C<medium_positive>,
C<strong_positive>, and C<abnormally_strong_positive>.

The categories into which the algorithm places numeric inputs act as
enumerated values.

=head1 PREREQUISITES

To use this module, you must have the MachineLearning module installed.

=head1 METHODS

=over

=item MachineLearning::DecisionTrees->new($args);

This is the constructor.

In addition to the class-name argument, which is passed automatically when
you use the C<MachineLearning::DecisionTrees-E<gt>new()> syntax, the
constructor takes a reference to a hash containing the following keys:

  InputFieldNames
  OutputFieldNames
  InputDataTypes
  TrainingData

The values associated with the InputFieldNames and OutputFieldNames keys
must each be a reference to an array of field names.  All field names (for
input and output fields combined) must be unique.  Field names must not
contain commas or line-break characters.  There must be at least two input
field names and at least one output field name.

The value associated with the InputDataTypes key must be a reference to a
hash in which the keys are input field names (which must match those
specified by the InputFieldNames argument) and each value indicates the
data type for the corresponding input field.  If the value is an array
reference, the field has a data type of C<enumeration> and the array must
contain two or more strings, all unique, representing the possible values
for the input field.  Otherwise, the value indicating the data type must be
the string C<number> or the string C<special_number>.

The value associated with the TrainingData key must be the path to a file
containing CSV-format training data.  The first line in the file must
contain field names, which must match the field names specified by the
InputFieldNames and OutputFieldNames arguments.

The constructor returns a reference to a MachineLearning::DecisionTrees
object (a I<grove>), which is implemented internally as a hash.  All
functionality is exposed through methods.

If the constructor receives a valid hash reference providing all required
information, the C<get_success()> instance method returns true (C<1>) and
the C<get_message()> instance method returns an empty string; otherwise,
C<get_success()> returns false (C<0>) and C<get_message()> returns a string
containing an error message.

=item $grove->get_success();

This returns true (C<1>) if the grove was initialized successfully;
otherwise, it returns false (C<0>).

=item $grove->get_message();

When C<get_success()> returns true (C<1>), this returns an empty string.
When C<get_success()> returns false (C<0>), C<get_message()> returns an
error message.

=item $grove->save($file_path);

This method saves (I<serializes>) the grove to the specified file.

If the serialization and file-writing operations succeed, this method
returns true (C<1>); otherwise, the method sets the C<_success> field
to false (C<0>), places an error message in the C<_message> field, and
returns false (C<0>).

=item MachineLearning::DecisionTrees->open($file_path);

This method returns a new MachinLearning::DecisionTrees object (a I<grove>)
created by restoring such an object from the specified file.  If the
file-reading and deserialization operations succeed, the resulting object's
C<get_success()> method returns true (C<1>) and the C<get_message()> method
returns an empty string.  Otherwise, the C<open()> method creates and
returns a new object that has a false value in the C<_success> field and an
error message in the C<_message> field.

=item $grove->print_out($tree_name);

This method prints out a particular tree within the grove.  The name that
you provide to the method is the name of the output field corresponding to
the tree that you want to study.

The printout lists each leaf node in order by quality (highest to lowest).
For each leaf node, the printout gives a number with which you can specify
the node when you call the C<test()> method or the C<employ()> method, the
ones percentage for the node after training, the number of records
associated with the node after training, and the path through the tree
leading to the node.

This method returns a string.

=item $grove->test($args);

This method takes a reference to a hash with the following fields:

    TreeName - The name of the tree, which is the same as the name
    of the output field for which the tree was built.

    ValidationData - The path to a file containing CSV-format
    validation data.  This must have the same fields and data types
    as the training data used to create the grove.

    NodeList - A reference to an array containing the numbers of the
    nodes that you want to use in the test to indicate "1" results.

This method returns a string giving the test results, which comprise the
number of records in the test data, the number of records assigned a "1"
result during the test, and the accuracy (as a percentage) of those "1"
results based on the validation data.

=item $grove->employ($args);

This method takes a reference to a hash with the following fields:

    TreeName - The name of the tree, which is the same as the name
    of the output field for which the tree was built.

    TargetData - The path to a file containing CSV-format data.
    This must have the same fields and data types as the training
    data used to create the grove.

    NodeList - A reference to an array containing the numbers of the
    nodes that you want to use to generate "1" results.

This method divides the records from the specified data file among the
leaf nodes of the specified tree, assigning a "1" result to any record
associated with one of the nodes given by the C<NodeList> argument.
For those records, the method places a C<1> in the output field
associated with the tree.  For the remaining records, the method places
a C<0> in that field.

The data file that you provide to this method must contain an output field
with the same name as the tree; however, the values in that field can be
blank.  If an output value is not blank, it must be a C<1> or a C<0> to
conform to the data type defined for output fields.  This method populates
or overwrites the values in the output field with new results.

If everything goes OK, this method returns true (C<1>); otherwise, the
method sets the C<_success> field to false (C<0>), places an error message
in the C<_message> field, and returns false (C<0>).

=back

=head1 AUTHOR

Benjamin Fitch, <blernflerkl@yahoo.com>

=head1 COPYRIGHT AND LICENSE

Copyright 2009 by Benjamin Fitch

This library is free software; you can redistribute it and/or modify it
under the same terms as Perl itself.

=cut
