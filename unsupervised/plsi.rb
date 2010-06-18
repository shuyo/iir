#!/usr/bin/ruby -KN
# ./plsi.rb [corpus files]

begin
  require '../lib/infinitive.rb'
  INF = Infinitive.new
rescue
  module INF
    def self.infinitive(word);word.downcase;end
  end
end

def dump(obj)
  if obj.is_a?(Numeric)
    (obj*1000).round/1000.0
  elsif obj.is_a?(String)
    obj
  elsif obj.is_a?(Array)
    "[#{obj.map{|x| dump(x)}.join(',')}]\n"
  elsif obj.is_a?(Hash)
    "{#{obj.map{|k,v| "#{k}=>#{dump(v)}"}.join(',')}}\n"
  end
end

docs = Array.new
words = Hash.new{|h,k| h[k]=Hash.new(0) }
worddocs = Hash.new{|h,k| h[k]=Hash.new(0) }
while filename = ARGV.shift
  puts "loading: #{filename}"
  texts = open(filename) {|f| f.read }.split(/\n\n+/)

  texts.each_with_index do |text, doc_id|
    vec = Hash.new(0)
    docs << vec
    text.scan(/[A-Za-z]+/) do |word|
      infword = INF.infinitive(word)
      vec[infword] += 1
      words[infword][word] += 1
      worddocs[infword][doc_id] += 1
    end
  end
end
puts "# of words = #{words.size}, # of docs = #{docs.length}"

class PLSI
  K = 20
  def initialize(docs, words, worddocs)
    @docs = docs
    @words = words
    @worddocs = worddocs
    
    @z_k = Array.new(K){1.0/K}
    @d_i_z_k = Array.new(K){ Array.new(docs.length){1.0/docs.length} }
    @w_j_z_k = Array.new(K){
      h = Hash.new
      s = 0
      worddocs.each{|j,x| s+=(h[j]=rand) }
      worddocs.each{|j,x| h[j]/=s }
      h
    }
  end

  def stepEM
    new_z_k_numer = Array.new(K){0}
    new_z_k_denom = 0
    new_d_i_numer = Array.new(K){ Array.new(@docs.length){0} }
    new_w_j_numer = Array.new(K){ Hash.new(0) }

    @worddocs.each do |j, n_w_j|
      #(0..@docs.length-1).each do |i|
      #n_w_j_d_i = n_w_j[i]
      n_w_j.each do |i, n_w_j_d_i|

        # E-step
        posterior_denom = 0
        posterior_numers = Array.new(K)
        (0..K-1).each do |k|
          # p(z=k)p(x|z)p(y|z)
          posterior_denom += (posterior_numers[k] = @z_k[k] * @d_i_z_k[k][i] * @w_j_z_k[k][j])
        end

        # M-step
        posterior_numers.each_with_index do |posterior_numer, k|
          x = n_w_j_d_i * posterior_numer / posterior_denom
          new_z_k_numer[k] += x
          new_d_i_numer[k][i] += x
          new_w_j_numer[k][j] += x
        end
        new_z_k_denom += n_w_j_d_i
      end
    end

    @z_k = new_z_k_numer.map{|x| x / new_z_k_denom }

    new_d_i_numer.each_with_index do |d_i, k|
      d_i.each_with_index do |numer, i|
        @d_i_z_k[k][i] = numer / new_z_k_numer[k]
      end
    end

    new_w_j_numer.each_with_index do |w_j, k|
      w_j.each do |j, numer|
        @w_j_z_k[k][j] = numer / new_z_k_numer[k]
      end
    end

    #puts "----"
    puts dump(@z_k)
    #puts dump(@d_i_z_k)
    #puts dump(@w_j_z_k)
  end

  def max_z_k_w_j
    cluster = Array.new(K){ Array.new }
    @worddocs.each do |j, n_w_j|
      argmax_k = nil
      max_z_k = 0
      sum = 0
      (0..K-1).each do |k|
        p_z_k_w_j = @z_k[k] * @w_j_z_k[k][j]
        sum += p_z_k_w_j
        if max_z_k < p_z_k_w_j
          max_z_k = p_z_k_w_j
          argmax_k = k
        end
      end
      cluster[argmax_k] << [j, max_z_k / sum]
    end
    cluster
  end
end

plsi = PLSI.new(docs, words, worddocs)
200.times{ plsi.stepEM }

cluster = plsi.max_z_k_w_j
cluster.each_with_index do |words, k|
  puts "  cluster: #{k}"
  sep = 1.0
  output = []
  words.sort_by{|x| -x[1] }.each do |x|
    while sep >= x[1]
      output << (sep*10).round
      sep -= 0.1
    end
    output << x[0]
  end
  puts output.join(',')
end

