####
# MachineLearning.pm:  A Perl module that provides a base class for
# MachineLearning objects.
#
# Copyright 2009 by Benjamin Fitch
#
# This library is free software; you can redistribute it and/or modify it
# under the same terms as Perl itself.
####
package MachineLearning;
use 5.008;
use strict;
use warnings;
use utf8;
use English "-no_match_vars";
use List::Util ("first", "sum");

# Version:
our $VERSION = '1.02';

# Constants:
my $POUND     = "#";
my $SQ        = "'";
my $DQ        = "\"";
my $SEMICOLON = ";";

my $INCLUDE_TRAILING_EMPTY_FIELDS = -1;

# Constructor:
sub new {
    my ($class, $args) = @_;
    my %ml_object;

    # Initialize the private fields:
    $ml_object{"_success"} = 1;
    $ml_object{"_message"} = "";
    $ml_object{"_input_field_names"} = [];
    $ml_object{"_output_field_names"} = [];

    # Parse the constructor arguments (other than the class name) and update
    # the private fields:
    if (ref($args) eq "HASH"
      && exists($args->{"InputFieldNames"})
      && exists($args->{"OutputFieldNames"})
      && ref($args->{"InputFieldNames"})  eq "ARRAY"
      && ref($args->{"OutputFieldNames"}) eq "ARRAY"
      && scalar(@{ $args->{"InputFieldNames"} }) >= 2
      && scalar(@{ $args->{"OutputFieldNames"} }) >= 1) {
        $ml_object{"_input_field_names"} = $args->{"InputFieldNames"};
        $ml_object{"_output_field_names"} = $args->{"OutputFieldNames"};
    }
    else {
        $ml_object{"_success"} = 0;
        $ml_object{"_message"} = "Invalid argument passed to the "
          . "$class constructor.";
    } # end if

    # Associate %ml_object with the MachineLearning class and then return a
    # reference to %ml_object:
    bless \%ml_object, $class;
} # end constructor

# Public instance methods:

sub get_success {
    return $_[0]->{"_success"};
} # end sub

sub get_message {
    return $_[0]->{"_message"};
} # end sub

# Public class methods:

# In addition to the class name passed in automatically, the get_mean()
# class method takes a reference to an array and returns the mean value
# for the set of numbers stored in the array.  When calling this function,
# make sure that the array contains at least one element and that each
# element is a number (for example:  3.9).
sub get_mean {
    shift; # Discard the class name
    my @nums = @{ $_[0] };
    my $n = scalar @nums;
    my $result = sum(@nums) / $n;

    return $result;
}

# In addition to the class name passed in automatically, the
# get_pop_std_dev() function takes a reference to an array and returns the
# population standard deviation for the set of numbers stored in the array.
# When calling this function, make sure that the array contains at least
# one element and that each element is a number (for example:  3.9).
sub get_pop_std_dev {
    my $class = $_[0];
    my @nums = @{ $_[1] };
    my $mean = $class->get_mean($_[1]);
    my $n = scalar @nums;
    my $result = sqrt(sum(map {($_ - $mean) ** 2} @nums) / $n);

    return $result;
}

# In addition to the class name passed in automatically, the
# get_samp_std_dev() function takes a reference to an array and returns the
# sample standard deviation for the set of numbers stored in the array.
# When calling this function, make sure that the array contains at least
# one element and that each element is a number (for example:  3.9).
sub get_samp_std_dev {
    my $class = $_[0];
    my @nums = @{ $_[1] };
    my $mean = $class->get_mean($_[1]);
    my $n = scalar @nums;
    my $result = sqrt(sum(map {($_ - $mean) ** 2} @nums) / ($n - 1));

    return $result;
}

# Private methods:

# The private _parse_data_file() instance method takes the following two
# arguments in addition to the object reference passed in automatically:
#
#  *  A path to a CSV file in which the column headings and field values
#     do not contain commas or line-break characters
#
#  *  A Boolean value indicating whether to allow blank values in the
#     output fields
#
# The method reads the contents of the CSV file into memory, strips double
# quotation marks, and then parses the data into an array of hashes in which
# the keys are column headings and the values are the corresponding field
# values for a particular record.  Each array element represents one record,
# where the records are separated by line breaks and do not include the
# header row.
#
#   IMPORTANT:  Only those fields with names that match the field names
#   specified for the object on which the method was called are included
#   in the returned data.
#
# While parsing the data, the method checks the validity of the both the
# overall structure of the data and the individual values.  The overall
# structure is valid if the number of column headings matches the number of
# values in each record, the column headings are all unique, there is a
# matching column heading for each input and output field specified for the
# object on which the method was called, and there are at least two records
# following the required header row.  The individual field names and values
# must not contain commas or line-break characters.
#
# If blank values are not allowed in the output fields, each output-field
# value must be a number equal to either 0 or 1.  If blank values are
# allowed in those fields, each output-field value must be either an
# empty string or a number equal to 0 or 1.  If an output-field value is
# an empty string, the method replaces it with a 0 in the returned data.
#
# The method strips any leading or trailing whitespace around a field name
# or value, converts tab characters to spaces, and then converts any string
# of consecutive spaces to a single space.
#
# The method returns a reference to an array containing records.  If the
# method cannot find or read the specified data file, or if the data is not
# valid, the method sets the calling object's "_success" field to 0, places
# an error message in the calling object's "_message" field, and returns a
# reference to an empty array.
#
# The _parse_data_file() method is for use only within the current module
# and any module associated with a class derived from the current class.
sub _parse_data_file {
    my $self = $_[0];
    my $csv_file_path = $_[1];
    my $allow_blank_values = $_[2];

    my $num_input_fields = scalar(@{ $self->{"_input_field_names"} });
    my $num_output_fields = scalar(@{ $self->{"_output_field_names"} });

    my $contents;
    my $header_row;
    my @field_names;
    my @body_rows;
    my @records;

    if (open my $CSV_FILE, $csv_file_path) {
        $contents = join "", <$CSV_FILE>;
        close $CSV_FILE;
    }
    else {
        $self->{"_success"} = 0;
        $self->{"_message"} = "Couldn't open $DQ$csv_file_path$DQ "
          . "for reading:\n\n    $!";
        return [];
    } # end if

    $contents = ref($self)->_normalize_csv_data($contents);

    unless (length $contents) {
        $self->{"_success"} = 0;
        $self->{"_message"} = "$DQ$csv_file_path$DQ contains invalid "
          . "CSV data, unsupported characters, duplicate column headings, "
          . "or an insufficient number of rows.";
        return [];
    } # end unless

    $contents = $self->_strip_extra_fields($contents);

    unless (length $contents) {
        $self->{"_success"} = 0;
        $self->{"_message"} = "$DQ$csv_file_path$DQ contains no fields "
          . "matching those specified for the " . ref($self) . " object.";
        return [];
    } # end unless

    @body_rows = split /[\r\n]+/s, $contents;
    $header_row = shift @body_rows;

    @field_names = split /,/s, $header_row, $INCLUDE_TRAILING_EMPTY_FIELDS;

    unless (
      scalar(@field_names) == $num_input_fields + $num_output_fields) {
        $self->{"_success"} = 0;
        $self->{"_message"} = "$DQ$csv_file_path$DQ does not contain all "
          . "the field names specified for the " . ref($self) . " object.";
        return [];
    } # end unless

    for my $field_name_index (0..$#field_names) {
        my $temp_index;
        my $temp_name;

        if ($field_name_index > $#{ $self->{"_input_field_names"} }) {
            $temp_index = $field_name_index
              - scalar(@{ $self->{"_input_field_names"} });
            $temp_name = $self->{"_output_field_names"}->[$temp_index];
        }
        else {
            $temp_index = $field_name_index;
            $temp_name = $self->{"_input_field_names"}->[$temp_index];
        } # end if

        unless ($field_names[$field_name_index] eq $temp_name) {
            $self->{"_success"} = 0;
            $self->{"_message"} = "The column headings in "
              . "$DQ$csv_file_path$DQ are not in the correct order.";
            return [];
        } # end unless
    } # next $field_name_index

    for my $body_row (@body_rows) {
        my @values = split /,/, $body_row, $INCLUDE_TRAILING_EMPTY_FIELDS;
        my $record = {};

        for my $dex (0..$#field_names) {
            if ($dex >= $num_input_fields && $allow_blank_values
              && length($values[$dex]) == 0) {
                $values[$dex] = 0;
            } # end if

            if ($dex >= $num_input_fields) {
                unless ($values[$dex] == 0 || $values[$dex] == 1) {
                    $self->{"_success"} = 0;
                    $self->{"_message"} = "An output-field value in "
                      . "$DQ$csv_file_path$DQ is not either 0 or 1.";
                    return [];
                } # end unless
            } # end if

            $record->{$field_names[$dex]} = $values[$dex];
        } # next $dex

        push @records, $record;
    } # next $body_row

    return \@records;
} # end sub

# In addition to the object reference passed in automatically, the private
# _strip_extra_fields() instance method takes a string containing the
# normalized contents of a CSV file.  The method returns a version of the
# string containing only fields with names that match the field names
# specified for the object on which the method was called.
sub _strip_extra_fields {
    my $self = $_[0];
    my $csv_data = $_[1];
    my @body_rows = split /[\r\n]+/s, $csv_data;
    my $header_row = shift @body_rows;
    my @column_headings = split /,/s, $header_row,
      $INCLUDE_TRAILING_EMPTY_FIELDS;
    my @required_field_names = (
      @{ $self->{"_input_field_names"} },
      @{ $self->{"_output_field_names"} });
    my @indices = ();

    for my $dex (0..$#column_headings) {
        if (defined first { $_ eq $column_headings[$dex]; }
          @required_field_names) {
            push @indices, $dex;
        } # end if
    } # next $dex

    for my $row ($header_row, @body_rows) {
        my @pieces = split /,/s, $row, $INCLUDE_TRAILING_EMPTY_FIELDS;

        $row = "";

        for my $dex (0..$#pieces) {
            if (defined first { $_ == $dex; } @indices) {
                $row .= $pieces[$dex] . ",";
            } # end if
        } # next $dex

        $row =~ s/,\z//s;
    } # next $row

    $csv_data = "";
    $csv_data .= $header_row . "\n";

    for my $body_row (@body_rows) {
        $csv_data .= $body_row . "\n";
    } # next $body_row

    return $csv_data;
} # end sub

# In addition to the class name passed in automatically, the private
# _normalize_csv_data() class method takes a string containing CSV data
# and returns a new version of the string after performing the following
# operations on it:
#
#  *  Strip double quotation marks
#
#  *  Convert tab characters to spaces
#
#  *  Strip any leading or trailing whitespace from field names and values
#
#  *  Replace any string of consecutive spaces with a single space
#
#  *  Make sure that there are at least three rows (including the
#     header row)
#
#  *  Make sure that there are no duplicate column headings
#
#  *  Make sure that all rows have the same number of fields
#
# If the rows don't all have the same number of fields, if there are
# fewer than three rows (including the header row), or if there are
# duplicate column headings, the method returns an empty string.
sub _normalize_csv_data {
    shift; # Discard the class name
    my $csv_data = $_[0];
    my $header_row;
    my @field_names;
    my @body_rows;

    $csv_data =~ s/$DQ//gs;
    $csv_data =~ s/\t/ /gs;
    $csv_data =~ s/(?:\A|(?<=\n)) +//gs;
    $csv_data =~ s/ +(?=\r|\n|\z)//gs;
    $csv_data =~ s/ +,/,/gs;
    $csv_data =~ s/, +/,/gs;
    $csv_data =~ s/ {2,}/ /gs;

    @body_rows = split /[\r\n]+/s, $csv_data;

    unless (scalar(@body_rows) >= 3) {
        return "";
    } # end unless

    $header_row = shift @body_rows;

    @field_names = split /,/s, $header_row, $INCLUDE_TRAILING_EMPTY_FIELDS;
    @field_names = sort @field_names;

    for my $dex (1..$#field_names) {
        if ($field_names[$dex - 1] eq $field_names[$dex]) {
            return "";
        } # end if
    } # next $dex

    for my $body_row (@body_rows) {
        my @pieces = split /,/s, $body_row, $INCLUDE_TRAILING_EMPTY_FIELDS;

        unless ($#pieces == $#field_names) {
            return "";
        } # end unless
    } # next $body_row

    return $csv_data;
} # end sub

# Module return value:
1;
__END__

=head1 NAME

MachineLearning - base class for MachineLearning objects

=head1 SYNOPSIS

  use base ("MachineLearning");

=head1 DESCRIPTION

This module provides a base class for MachineLearning objects.

In addition to the class name passed in automatically when you use the
C<$class-E<gt>SUPER::new($args)> syntax, the constructor takes a reference
to a hash containing the following keys:

  InputFieldNames
  OutputFieldNames

The value corresponding to each of these keys must be a reference to an
array of field names.  All field names (for input and output fields
combined) must be unique.  Field names must not contain commas or line-break
characters.  There must be at least two input field names and at least one
output field name.

In addition to a base constructor, this super class provides the following
fields and instance methods:

  _success
  _message
  _input_field_names
  _output_field_names
  _parse_data_file()
  get_success()
  get_message()

This class also provides the following class methods:

  get_mean()
  get_pop_std_dev()
  get_samp_std_dev()

In addition to the class name passed in automatically, each of these class
methods takes a reference to an array and returns the appropriate value
(mean, population standard deviation, or sample standard deviation) for the
set of numbers stored in the array.  When calling one of these functions,
make sure that the array contains at least one element and that each element
is a number (for example:  3.9).

=head1 AUTHOR

Benjamin Fitch, <blernflerkl@yahoo.com>

=head1 COPYRIGHT AND LICENSE

Copyright 2009 by Benjamin Fitch

This library is free software; you can redistribute it and/or modify it
under the same terms as Perl itself.

=cut
