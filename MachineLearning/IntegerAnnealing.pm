####
# IntegerAnnealing.pm:  A Perl module that contains a single public
# function, anneal(), which optimizes a list of integers according to a
# cost function.
#
# Copyright 2009 by Benjamin Fitch
#
# This library is free software; you can redistribute it and/or modify it
# under the same terms as Perl itself.
####
package MachineLearning::IntegerAnnealing;
use 5.008;
use strict;
use warnings;
use utf8;

use English "-no_match_vars";
use Scalar::Util ("looks_like_number");
use POSIX ("ceil", "floor");
use Exporter;

# Version:
our $VERSION = '1.03';

# Exports:
our @ISA = ("Exporter");
our @EXPORT = ("anneal");

# Constants:
my $POUND     = "#";
my $SQ        = "'";
my $DQ        = "\"";
my $SEMICOLON = ";";
my $SPACE     = " ";
my $EMPTY     = "";
my $TRUE      = 1;
my $FALSE     = 0;

my $TEMPERATURE_REDUCTION_MULTIPLIER = 0.96;

# Public functions:

sub anneal {
    my $ranges = $_[0]->{"Ranges"};
    my $cost_calculator = $_[0]->{"CostCalculator"};
    my $cycles_per_temperature = $_[0]->{"CyclesPerTemperature"};

    my @result_array;
    my @current_array;
    my $current_cost;
    my $current_temperature;

    if (looks_like_number($cycles_per_temperature)
      && int($cycles_per_temperature) > 0) {
        $cycles_per_temperature = int $cycles_per_temperature;
    }
    else {
        return [];
    } # end if

    # Initialize @current_array using the midpoint of each range as the
    # starting values:
    for my $range (@{ $ranges }) {
        push @current_array, ceil(($range->[0] + $range->[1]) / 2);
    } # next $range

    # Get the cost of the starting array:
    $current_cost = $cost_calculator->(\@current_array);

    # Start with a temperature equal to the size of the largest range, and
    # proceed with the annealing:
    $current_temperature = $ranges->[0]->[1] - $ranges->[0]->[0];

    for my $dex (1..$#{ $ranges }) {
        my $contender = $ranges->[$dex]->[1] - $ranges->[$dex]->[0];

        if ($contender > $current_temperature) {
            $current_temperature = $contender;
        } # end if
    } # next $dex

    while ($current_temperature > 0) {
        for (1..$cycles_per_temperature) {
            my @temp_array;
            my $temp_cost;

            for my $dex (0..$#{ $ranges }) {
                my $ger = $current_array[$dex];
                my $lower_bound = $ranges->[$dex]->[0];
                my $upper_bound = $ranges->[$dex]->[1];
                my $range_size = $upper_bound - $lower_bound;
                my $new_lower_bound = $lower_bound;
                my $new_upper_bound = $upper_bound;
                my $new_range_size = $range_size;
                my $chosen_integer = $ger;

                unless ($range_size <= $current_temperature) {
                    $new_lower_bound = $ger - ($current_temperature / 2);
                    $new_upper_bound = $ger + ($current_temperature / 2);

                    if ($new_lower_bound < $lower_bound) {
                        my $diff = $lower_bound - $new_lower_bound;

                        $new_lower_bound = $lower_bound;
                        $new_upper_bound += $diff;
                    }
                    elsif ($new_upper_bound > $upper_bound) {
                        my $diff = $new_upper_bound - $upper_bound;

                        $new_upper_bound = $upper_bound;
                        $new_lower_bound -= $diff;
                    } # end if

                    $new_lower_bound = ceil($new_lower_bound);
                    $new_upper_bound = floor($new_upper_bound);
                    $new_range_size = $new_upper_bound - $new_lower_bound;
                } # end unless

                unless ($new_range_size == 0) {
                    $chosen_integer = _choose_integer(
                      $new_lower_bound, $new_upper_bound);
                } # end unless

                push @temp_array, $chosen_integer;
            } # next $dex

            $temp_cost = $cost_calculator->(\@temp_array);

            if ($temp_cost < $current_cost) {
                @current_array = @temp_array;
                $current_cost = $temp_cost;
            } # end if
        } # next cycle

        $current_temperature = floor(
          $current_temperature * $TEMPERATURE_REDUCTION_MULTIPLIER);
    } # end while

    @result_array = @current_array;
    return \@result_array;
} # end sub

# Private functions:

# The _choose_integer() function takes a lower bound and an upper bound
# (two integers of which the first is less than the second), and returns
# a random integer that is greater than or equal to the lower bound and
# less than or equal to the upper bound.
sub _choose_integer {
    my $lower_bound = $_[0];
    my $upper_bound = $_[1];
    my $range_size = $upper_bound - $lower_bound;
    my $random_integer = int(rand($range_size + 1));
    my $chosen_integer = $lower_bound + $random_integer;

    return $chosen_integer;
} # end sub

# Module return value:
1;
__END__

=head1 NAME

MachineLearning::IntegerAnnealing - optimize a list of integers according to a cost function

=head1 SYNOPSIS

  use MachineLearning::IntegerAnnealing;
  my $result_array_ref = anneal({
    "Ranges" => [ [2, 4], [9, 18], [19, 28], [29, 55] ],
    "CostCalculator" => $cost_calculator_coderef,
    "CyclesPerTemperature" => 1_000});

=head1 DESCRIPTION

This module exports a single function, C<anneal()>, which performs simulated
annealing to optimize a list of integers that have predefined ranges.  The
list can be of any size C<N>.

In addition to the ranges for the integers, the C<anneal()> function takes
a reference to a cost function that takes a reference to an array with C<N>
elements and returns a number representing a cost to be minimized.  The
returned number does not have to be an integer.

The C<anneal()> function also takes as input a positive integer specifying
the number of cycles per temperature; that is, the number of randomization
cycles to perform at each temperature level during the annealing process.
A higher number of cycles per temperature produces more accurate results
while increasing the amount of time required for the annealing process
to complete.

=head1 FUNCTIONS

=over

=item anneal($args_hashref);

This function takes a reference to a hash with the following fields:

    Ranges - A reference to an array of pairs of bounds, lower and
    upper, where a pair is a reference to an array of two integers
    of which the first is less than the second.

    CostCalculator - A reference to a function that takes a
    reference to an array of integers and returns a single number
    representing a cost to be minimized.  The function must accept
    a reference to an input array that is the same size as the
    Ranges array.

      NOTE:  The returned number does not have to be an integer.

    CyclesPerTemperature - A positive integer specifying the number
    of randomization cycles performed at each temperature level.

      NOTE:  Temperature starts at the size of the largest range
      (which means that each integer gets randomized within 100% of
      its specified range) and then gradually decreases.  Each
      temperature reduction multiplies the temperature by 96% and
      then rounds that result down to the nearest integer.

    If the CyclesPerTemperature value is not a positive integer,
    the anneal() function returns a reference to an empty array.

The C<anneal()> function returns a reference to an array of integers that
corresponds to the Ranges array (that is, the output array is the same size
as the Ranges array, and each integer in the output array is within the
range indicated by the corresponding element in the Ranges array).  The
output array is the list of integers that has the lowest cost (according to
the specified cost function) of any of the lists tested during the annealing
process.

=back

=head1 AUTHOR

Benjamin Fitch, <blernflerkl@yahoo.com>

=head1 COPYRIGHT AND LICENSE

Copyright 2009 by Benjamin Fitch

This library is free software; you can redistribute it and/or modify it
under the same terms as Perl itself.

=cut
