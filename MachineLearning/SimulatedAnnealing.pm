####
# SimulatedAnnealing.pm:  A Perl module that contains a single public
# function, anneal(), which optimizes a list of numbers according to a
# cost function.
#
# Copyright 2009 by Benjamin Fitch
#
# This library is free software; you can redistribute it and/or modify it
# under the same terms as Perl itself.
####
package MachineLearning::SimulatedAnnealing;
use 5.008;
use strict;
use warnings;
use utf8;
use English "-no_match_vars";
use Scalar::Util ("looks_like_number");
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

    # Initialize @current_array using the exact midpoint of each range as
    # the starting values:
    for my $range (@{ $ranges }) {
        push @current_array, ($range->[0] + $range->[1]) / 2;
    } # next $range

    # Get the cost of the starting array:
    $current_cost = $cost_calculator->(\@current_array);

    # Start with a temperature of 100, and proceed with the annealing:
    $current_temperature = 100;

    while ($current_temperature > 0) {
        for (1..$cycles_per_temperature) {
            my @temp_array;
            my $temp_cost;

            for my $dex (0..$#{ $ranges }) {
                my $num = $current_array[$dex];
                my $lower_bound = $ranges->[$dex]->[0];
                my $upper_bound = $ranges->[$dex]->[1];
                my $range_size = $upper_bound - $lower_bound;
                my $sub_range_size
                  = $range_size * ($current_temperature / 100);
                my $sub_lower_bound = $num - ($sub_range_size / 2);
                my $sub_upper_bound = $num + ($sub_range_size / 2);

                if ($sub_lower_bound < $lower_bound) {
                    $sub_lower_bound = $lower_bound;
                    $sub_upper_bound = $sub_lower_bound + $sub_range_size;
                }
                elsif ($sub_upper_bound > $upper_bound) {
                    $sub_upper_bound = $upper_bound;
                    $sub_lower_bound = $sub_upper_bound - $sub_range_size;
                } # end if

                push @temp_array, _choose_number(
                  $sub_lower_bound, $sub_upper_bound);
            } # next $dex

            $temp_cost = $cost_calculator->(\@temp_array);

            if ($temp_cost < $current_cost) {
                @current_array = @temp_array;
                $current_cost = $temp_cost;
            } # end if
        } # next cycle

        $current_temperature = _reduce_temperature($current_temperature);
    } # end while

    @result_array = @current_array;
    return \@result_array;
} # end sub

# Private functions:

# The _choose_number() function takes a lower bound and an upper bound
# (real decimal numbers of which the first is less than the second), and
# returns a random number that is greater than or equal to the lower bound
# and less than or equal to the upper bound.
sub _choose_number {
    my $lower_bound = $_[0];
    my $upper_bound = $_[1];
    my $range_size = $upper_bound - $lower_bound;
    my $random_number = int(rand 100_000_001) / 100_000_000;
    my $new_number = ($random_number * $range_size) + $lower_bound;

    return $new_number;
} # end sub

# The _reduce_temperature() function takes a positive number not greater
# than 100 representing a temperature in the form of a percentage, and
# returns a number representing a reduced temperature.
#
#   IMPORTANT:  Pass to this function only the value 100 or a non-zero value
#   previously returned by this function.
sub _reduce_temperature {
    my $input = $_[0];

    return $input > 40.00 ? ($input - 5.000)
         : $input > 20.00 ? ($input - 2.000)
         : $input > 10.00 ? ($input - 1.000)
         : $input >  4.00 ? ($input - 0.500)
         : $input >  2.00 ? ($input - 0.200)
         : $input >  1.00 ? ($input - 0.100)
         : $input >  0.40 ? ($input - 0.050)
         : $input >  0.20 ? ($input - 0.020)
         : $input >  0.10 ? ($input - 0.010)
         : $input >  0.04 ? ($input - 0.005)
         : $input >  0.02 ? ($input - 0.002)
         :                  ($input - 0.001);
} # end sub

# Module return value:
1;
__END__

=head1 NAME

MachineLearning::SimulatedAnnealing - optimize a list of numbers according to a cost function

=head1 SYNOPSIS

  use MachineLearning::SimulatedAnnealing;
  my $result_array_ref = anneal({
    "Ranges" => [ [0, 3], [-1, 4], [-4, 0] ],
    "CostCalculator" => $cost_calculator_coderef,
    "CyclesPerTemperature" => 10_000});

=head1 DESCRIPTION

This module exports a single function, C<anneal()>, which performs simulated
annealing to optimize a list of numbers that have predefined ranges.  The
list can be of any size C<N>.

In addition to the ranges for the numbers, the C<anneal()> function takes
a reference to a cost function that takes a list of size C<N> and returns
a number representing a cost to be minimized.

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
    upper, where a pair is a reference to an array of two real
    decimal numbers of which the first is less than the second.

    CostCalculator - A reference to a function that takes a
    reference to an array of numbers and returns a single number
    representing a cost to be minimized.  The function must take an
    input array that is the same size as the Ranges array.

    CyclesPerTemperature - A positive integer specifying the number
    of randomization cycles performed at each temperature level.

      NOTE:  Temperature starts at 100% (which means that each
      number gets randomized within 100% of its specified range) and
      then gradually decreases.  To randomize a number within a
      range corresponding to a temperature that is less than 100%,
      the function calculates the appropriate size for the sub-range
      and then chooses the exact location for that sub-range (within
      the total range specified for the number) such that the
      current value of the number is as close as possible to the
      sub-range center.

      The temperature, expressed as a percentage, decreases as
      follows:

          100.000, 95.000, 90.000, 85.000, 80.000, 75.000, 70.000,
           65.000, 60.000, 55.000, 50.000, 45.000, 40.000, 38.000,
           36.000, 34.000, 32.000, 30.000, 28.000, 26.000, 24.000,
           22.000, 20.000, 19.000, 18.000, 17.000, 16.000, 15.000,
           14.000, 13.000, 12.000, 11.000, 10.000, 09.500, 09.000,
           08.500, 08.000, 07.500, 07.000, 06.500, 06.000, 05.500,
           05.000, 04.500, 04.000, 03.800, 03.600, 03.400, 03.200,
           03.000, 02.800, 02.600, 02.400, 02.200, 02.000, 01.900,
           01.800, 01.700, 01.600, 01.500, 01.400, 01.300, 01.200,
           01.100, 01.000, 00.950, 00.900, 00.850, 00.800, 00.750,
           00.700, 00.650, 00.600, 00.550, 00.500, 00.450, 00.400,
           00.380, 00.360, 00.340, 00.320, 00.300, 00.280, 00.260,
           00.240, 00.220, 00.200, 00.190, 00.180, 00.170, 00.160,
           00.150, 00.140, 00.130, 00.120, 00.110, 00.100, 00.095,
           00.090, 00.085, 00.080, 00.075, 00.070, 00.065, 00.060,
           00.055, 00.050, 00.045, 00.040, 00.038, 00.036, 00.034,
           00.032, 00.030, 00.028, 00.026, 00.024, 00.022, 00.020,
           00.019, 00.018, 00.017, 00.016, 00.015, 00.014, 00.013,
           00.012, 00.011, 00.010, 00.009, 00.008, 00.007, 00.006,
           00.005, 00.004, 00.003, 00.002, 00.001, 00.000

    If the CyclesPerTemperature value is not a positive integer,
    the anneal() function returns a reference to an empty array.

The C<anneal()> function returns a reference to an array of numbers that
corresponds to the Ranges array (that is, the output array is the same
size as the Ranges array, and each number in the output array is within
the range specified by the corresponding element in the Ranges array).
The output array is the list of numbers that has the lowest cost
(according to the specified cost function) of any of the lists tested
during the annealing process.

=back

=head1 AUTHOR

Benjamin Fitch, <blernflerkl@yahoo.com>

=head1 COPYRIGHT AND LICENSE

Copyright 2009 by Benjamin Fitch

This library is free software; you can redistribute it and/or modify it
under the same terms as Perl itself.

=cut
