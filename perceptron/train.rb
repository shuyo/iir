#!/usr/bin/ruby

require 'optparse'
opt = {:algo=>:P, :iteration=>10}
parser = OptionParser.new
parser.banner = "Usage: #$0 [options] trainfile modelfile"
parser.on('-i [VAL]', Integer, 'number of iteration') {|v| opt[:iteration] = v }
parser.on('-a [VAL]', [:P, :AP], 'algorism') {|v| opt[:algo] = v }
parser.parse!(ARGV)
if ARGV.length < 2
  $stderr.puts parser
  exit(1)
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

# perceptron
w = Array.new(degree + 1, 0)
opt[:iteration].times do |c|
  traindata = traindata.sort_by{rand} # shuffle

  # training
  n_errors = 0
  w_a = Array.new(degree + 1, 0) # for average perceptron
  n = 0
  traindata.each do |x, t|
    x[degree] = 1 # bias
    s = 0        # sigma w^T phai(x_n)
    x.each do |i, x_i|
      s += w[i] * x_i
    end
    if s * t <= 0  # error
      n_errors += 1
      x.each do |i, x_i|
        w[i] += t * x_i
        w_a[i] += t * x_i * n # for average perceptron
      end
    end
    n += 1
  end
  w_a.each_with_index do |w_i, i|
    w[i] -= w_i.to_f / n
  end if  opt[:algo] == :AP

  if n_errors == 0
    puts "convergence: #{c}"
    break
  end
end

open(ARGV[1], 'w'){|f| Marshal.dump(w, f) }

