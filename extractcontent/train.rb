#!/usr/bin/ruby -Ku

require 'optparse'
opt = {:algo=>:PA, :iteration=>10, :regularity=>1.0}
parser = OptionParser.new
parser.banner = "Usage: #$0 [options] trainfile modelfile"
parser.on('-i [VAL]', Integer, 'number of iteration') {|v| opt[:iteration] = v }
parser.on('-a [VAL]', [:PA, :PA1, :PA2], 'algorism(PA/PA1/PA2)') {|v| opt[:algo] = v }
parser.on('-C [VAL]', Float, 'regularization parameter (for PA1/PA2)') {|v| opt[:regularity] = v }
parser.parse!(ARGV)
if ARGV.length < 2
  $stderr.puts parser
  exit(1)
end


class Train
  def initialize(degree)
    @degree = degree
    @w = Array.new(degree + 1, 0)
  end
  attr_accessor :w

  def loop(traindata, docfreq_inv, iteration=1, will_shuffle=true)
    pre_w = @w.dup
    iteration.times do |c|

      traindata = traindata.sort_by{rand} if will_shuffle
      traindata.each do |x0, t|
        predict_cat_id = nil
        predict_value = -1
        docfreq_inv.each do |cat_id, docfreqs_except_cat|
          x = x0.dup
          docfreqs_except_cat.each do |term_id, log_inv|
            x[term_id] *= log_inv if x.key?(term_id)
          end
          x[@degree] = 1 # bias
          s = 0        # sigma w^T phai(x_n)
          x.each do |i, x_i|
            s += @w[i] * x_i
          end
          p [x, s]
          if predict_value < s
            predict_value = s
            predict_cat_id = cat_id
          end
        end
        p [x0, t, predict_cat_id, predict_value]
        break
        #yield @w, x, t, s
      end

      return c if pre_w == @w
    end
    nil
  end
end

def passive_aggressive(traindata, docfreq_inv, degree, iteration, aggressiveness=nil, regularity=0)
  training = Train.new(degree)
  c = training.loop(traindata, docfreq_inv, iteration) do |w, x, correct, predict|
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
categories = []
docfreq = Hash.new(0)
docfreq_each = Hash.new
degree = 0
open(ARGV[0]) do |f|
  while line = f.gets
    features = line.split

    cat = features.shift
    cat_id = categories.index(cat)
    if !cat_id
      cat_id = categories.length
      categories << cat
    end

    map = Hash.new
    features.each do |feature|
      if feature =~ /^([0-9]+):([\+\-]?[0-9\.]+)$/
        term_id = $1.to_i
        map[term_id] = $2.to_f
        degree = term_id + 1 if degree <= term_id
        docfreq[term_id] += 1
        docfreq_each[term_id] ||= Hash.new(0)
        docfreq_each[term_id][cat_id] += 1
      end
    end
    traindata << [map, cat_id]
  end
end
docfreq_inv = Hash.new
doc_num = traindata.length
categories.length.times do |cat_id|
  docfreq_inv[cat_id] = Hash.new
  docfreq.each do |term_id, freq|
    freq_except_cat = (freq - docfreq_each[term_id][cat_id])
    freq_except_cat = 0.5 if freq_except_cat <= 0
    docfreq_inv[cat_id][term_id] = Math.log( doc_num / freq_except_cat )
  end
end

training, convergence = passive_aggressive(traindata, docfreq_inv, degree, opt[:iteration])
puts "convergence: #{convergence}" if convergence
open(ARGV[1], 'w'){|f| Marshal.dump(training.w, f) }


