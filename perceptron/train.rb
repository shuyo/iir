#!/usr/bin/ruby

require 'optparse'
opt = {:algo=>:P, :iteration=>10, :regularity=>1.0}
parser = OptionParser.new
parser.banner = "Usage: #$0 [options] trainfile modelfile"
parser.on('-i [VAL]', Integer, 'number of iteration') {|v| opt[:iteration] = v }
parser.on('-a [VAL]', [:P, :AP, :PA, :PA1, :PA2], 'algorism(P/AP/PA/PA1/PA2)') {|v| opt[:algo] = v }
parser.on('-C [VAL]', Float, 'regularization parameter (for PA1/PA2)') {|v| opt[:regularity] = v }
parser.parse!(ARGV)
if ARGV.length < 2
  $stderr.puts parser
  exit(1)
end


# common

def square_abs(x)
  square_abs_x = 0
  x.each do |i, x_i|
    square_abs_x += x_i * x_i
  end
  square_abs_x
end

class Train
  def initialize(degree)
    @degree = degree
    @w = Array.new(degree + 1, 0)
  end
  attr_accessor :w

  def loop(traindata, iteration=1, will_shuffle=true)
    pre_w = @w.dup
    iteration.times do |c|

      traindata = traindata.sort_by{rand} if will_shuffle
      traindata.each do |x, t|
        x[@degree] = 1 # bias
        s = 0        # sigma w^T phai(x_n)
        x.each do |i, x_i|
          s += @w[i] * x_i
        end
        yield @w, x, t, s
      end

      return c if pre_w == @w
    end
    nil
  end
end


# algorism

def perceptron(traindata, degree, iteration)
  training = Train.new(degree)
  c = training.loop(traindata, iteration) do |w, x, t, s|
    if s * t <= 0  # error
      x.each do |i, x_i|
        w[i] += t * x_i
      end
    end
  end
  return [training, c]
end

def average_perceptron(traindata, degree, iteration)
  training = Train.new(degree)
  iteration.times do |c|
    w_a = Array.new(degree + 1, 0) # for average perceptron
    n = 0
    is_convergenced = training.loop(traindata) do |w, x, t, s|
      if s * t <= 0  # error
        x.each do |i, x_i|
          w[i] += t * x_i
          w_a[i] += t * x_i * n # for average perceptron
        end
      end
      n += 1
    end
    return [training, c] if is_convergenced
    w_a.each_with_index do |w_i, i|
      training.w[i] -= w_i.to_f / n     # for averate perceptron
    end
  end
  return [training, nil]
end

def passive_aggressive(traindata, degree, iteration, aggressiveness=nil, regularity=0)
  training = Train.new(degree)
  c = training.loop(traindata, iteration) do |w, x, correct, predict|
    loss = 1 - correct * predict
    if loss > 0
      tau = loss.to_f / (square_abs(x) + regularity)
      tau = aggressiveness if aggressiveness && tau > aggressiveness
      x.each do |i, x_i|
        w[i] += tau * correct * x_i
      end
    end
  end
  return [training, c]
end



# load training data
traindata = []
degree = 0
open(ARGV[0]) do |f|
  while line = f.gets
    features = line.split
    sign = features.shift.to_i
    map = Hash.new
    features.each do |feature|
      if feature =~ /^([0-9]+):([\+\-]?[0-9\.]+)$/
        term_id = $1.to_i
        map[term_id] = $2.to_f
        degree = term_id + 1 if degree <= term_id
      end
    end
    traindata << [map, sign]
  end
end

# training

training, convergence = if opt[:algo] == :P
  perceptron(traindata, degree, opt[:iteration])
elsif opt[:algo] == :AP
  average_perceptron(traindata, degree, opt[:iteration])
elsif opt[:algo] == :PA
  passive_aggressive(traindata, degree, opt[:iteration])
elsif opt[:algo] == :PA1
  passive_aggressive(traindata, degree, opt[:iteration], opt[:regularity])
elsif opt[:algo] == :PA2
  passive_aggressive(traindata, degree, opt[:iteration], nil, 0.5 / opt[:regularity])
end

puts "convergence: #{convergence}" if convergence
open(ARGV[1], 'w'){|f| Marshal.dump(training.w, f) }

