#!/usr/bin/ruby

class Unit
  def initialize(name=nil)
    @name = name
  end
  def name
    if @name then @name else super.to_s end
  end
  def ==(other)
    __id__==other.__id__
  end
end
class BiasUnit < Unit
end
class IdentityUnit < Unit
  def formula_name; ""; end
  def activation_func(a)
    a
  end
  def divback(z)
    1
  end
end
class TanhUnit < Unit
  def formula_name; "tanh"; end
  def activation_func(a)
    Math.tanh(a)
  end
  # divergence for backward
  def divback(z)
    1-z*z
  end
end
class SigUnit < Unit
  def formula_name; "sig"; end
  def activation_func(a)
    1.0/(1+Math.exp(-a))
  end
  def divback(z)
    z*(1-z)
  end
end


# weight parameters
class Weights
  EPSILON = 0.001
  def initialize
    @orig_parameters = @parameters = []
    @from_units = []
    @to_units = []
  end
  def append(from, to)
    @parameters << normrand(0, 1)
    @from_units << from
    @to_units << to
  end
  def in_units(out_list)
    in_list = Hash.new
    @to_units.each_with_index do |unit, i|
      in_list[@from_units[i]] = @parameters[i] if out_list == unit
    end
    in_list
  end
  def normrand(m=0, s=1)
    r=0
    12.times{ r+=rand() }
    (r - 6) * s + m
  end
  def plus_epsilon(index)
    @parameters = @orig_parameters.dup
    @parameters[index] += EPSILON
  end
  def minus_epsilon(index)
    @parameters = @orig_parameters.dup
    @parameters[index] -= EPSILON
  end
  def back_to_orig
    @parameters = @orig_parameters
  end
  def size
    @parameters.length
  end
  def descent(eta, grad)
    @parameters = []
    @orig_parameters.each_with_index do |w_i, i|
      @parameters << w_i - eta * grad[i]
    end
    @orig_parameters = @parameters
  end
  def parameters
    @parameters
  end

  def dump
    d = Hash.new
    @to_units.each_with_index do |unit, i|
      d[unit] ||= []
      d[unit] << "#{@parameters[i]} * #{@from_units[i].name}"
    end
    d.each do |unit, formula|
      puts "#{unit.name} <- #{unit.formula_name}( #{formula.join(" + ")} );"
    end
  end
end


# neural network
class Network
  def initialize
    @units = []
    @weights = Weights.new
    @in_list = []
    @out_list = []
    @forward_prop = nil
    @backward_prop = nil
  end
  def link(from_list, to_list)
    append_unit from_list
    append_unit to_list
    from_list.each do |from|
      to_list.each do |to|
        @weights.append from, to
      end
    end
  end
  def in=(in_list)
    @in_list = in_list
    append_unit in_list
  end
  def append_unit(list)
    list.each do |unit|
      @units << unit unless @units.include?(unit)
    end
  end
  def out=(out_list)
    @out_list = out_list
    out_list.each do |unit|
      raise "There is a output unit without link." unless @units.include?(unit)
    end
  end

  def arrange_units
    calcurated = Hash.new
    @units.each do |unit|
      calcurated[unit] = 1 if unit.instance_of?(BiasUnit) || @in_list.include?(unit)
    end

    arranged = []
    while calcurated.size < @units.length
      advance = false
      @units.each do |unit|
        next if calcurated.key?(unit)
        in_list = @weights.in_units(unit)
        if in_list.keys.all?{|z| calcurated.key?(z)}
          arranged << unit
          calcurated[unit] = 1
          advance = true
        end
      end
      raise "There is a not-calcurable unit." unless advance
    end

    arranged
  end

  def forward_prop
    @forward_prop = arrange_units unless @forward_prop
    @forward_prop
  end

  def apply(*params)
    raise "not equal # of parameters to # of input units" if params.length != @in_list.length

    values = Hash.new
    @units.each do |unit|
      values[unit] = 1 if unit.instance_of?(BiasUnit)
    end
    @in_list.each_with_index do |unit, i|
      values[unit] = params[i]
    end

    forward_prop.each do |unit|
      linear_units = @weights.in_units(unit)
      a = 0
      linear_units.each do |z, w|
        a += w * values[z]
      end
      values[unit] = unit.activation_func(a)
    end
    #values.each {|unit, z| puts "#{unit.name} = #{z}" }

    @out_list.map{|unit| values[unit]}
  end

  def sum_of_squares_error(x, t)
    y = apply(*x)
    e = 0
    y.each_with_index do |y_i, i|
      e += (y_i - t[i]) ** 2
    end
    e / 2
  end

  # divergence with central difference
  def divergent_E(index, x, t)
    @weights.plus_epsilon index
    e1 = sum_of_squares_error(x, t)
    @weights.minus_epsilon index
    e2 = sum_of_squares_error(x, t)

    (e1 - e2) / (2 * Weights::EPSILON)
  end
  
  def gradient_E(x, t)
    g = []
    @weights.size.times do |i|
      g << divergent_E(i, x, t)
    end
    g
  end

  def descent_weights(eta, grad)
    @weights.descent(eta, grad)
  end

  def weights
    @weights
  end
end


