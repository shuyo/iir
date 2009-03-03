#!/usr/bin/ruby

DEBUG = false

# tf-idf
def calc_tdidf(tf, df, n_docs)
  tf * Math.log(n_docs.to_f / df)
end

# inner product
def calc_sim(v1, v2)
  sim = 0
  v1.each_with_index {|x, i| sim += x * v2[i] }
  sim
end

class SymmetricHash
  def initialize
    @map = Hash.new
  end
  def [](d1, d2)
    @map[[d1, d2]] || @map[[d2, d1]]
  end
  def []=(d1, d2, v)
    if @map[key = [d2, d1]]
      @map[key] = v
    else
      @map[[d1, d2]] = v
    end
  end
end

# うそプライオリティキュー
class PriorityQueue
  def initialize
    clear
  end
  def clear; @array = Array.new; end
  def max; @array[0]; end
  def insert(x)
    i = 0
    while i<@array.length
      break if x[0]>@array[i][0]
      i+=1
    end
    @array.insert(i, x)
  end
  def delete_if_3rd(x2)
    @array.delete_if{|x| x[2]==x2 }
  end
end

# draw dendrogram
class Dendrogram
  def initialize(clusters)
    @tree = Hash.new

    @lines = Array.new
    @index = Hash.new
    @rank = 0
  end
  def get_tree(d)
    if @tree.key?(d)
      @tree.delete(d)
    else
      d
    end
  end
  def branch(d1, d2)
    @tree[d1] = [get_tree(d1), get_tree(d2)]
  end

  def gen_lines(branch)
    if branch.instance_of?(Array)
      gen_lines branch[0]
      gen_lines branch[1]
    else
      @lines << "- " + branch
      @index[branch] = @lines.size - 1
    end
  end

  def gen_output(clusters)
    gen_lines @tree.values[0]
    clusters.each do |d1, d2, s|
      connect d1, d2
    end
  end
  def get_index(d)
    return @index[d] if @index.key?(d)
    @lines << "#{'-' * @rank} #{d}"
    @index[d] = @lines.size - 1
  end
  def connect(d1, d2)
    i1 = @index[d1]
    i2 = @index[d2]
    i1, i2 = i2, i1 if i2 < i1
    @lines.each_with_index do |line, i|
      if i==i1
        @lines[i] = "-" + line
      elsif i==i2
        @lines[i] = "+" + line
      elsif i>i1 && i<i2
        @lines[i] = "|" + line
      elsif line =~ /^-/
        @lines[i] = "-" + line
      else
        @lines[i] = " " + line
      end
    end
    @rank += 1
  end
  def to_s
    @lines.join("\n")
  end
end


data = open(ARGV[0] || 'corpus'){|f| Marshal.load(f) }
docs = data[:docs]
terms = data[:terms]

# TODO: feature selection
axes = terms.keys # all terms
#axes = terms.keys.select{|term| n=terms[term].size; n>1 && n<docs.size } # 2～n_docs-1

# 特徴ベクトルの計算
docs.each_with_index do |doc, doc_id|
  l2 = 0
  v = axes.map do |term|
    rev_index = terms[term]
    docfreq = rev_index.size
    termfreq = rev_index[doc_id]
    x = calc_tdidf(termfreq, docfreq, docs.size)
    l2 += x * x
    x
  end
  l = Math.sqrt(l2)
  doc[:vector] = v.map{|x| x / l}
end


def naive_hac(docs)
  # similarity & priority queue
  sim = SymmetricHash.new
  docs.each do |d2|
    docs.each do |d1|
      break if d1 == d2
      sim[d1[:title], d2[:title]] = v = calc_sim(d1[:vector], d2[:vector])
      p [d1[:title], d2[:title], v] if DEBUG
    end
  end

  puts "---" if DEBUG

  # hac
  merged = []
  while docs.size > 1
    maxsim = 0
    maxsim_pair = nil
    docs.each do |d2|
      docs.each do |d1|
        break if d1==d2
        s = sim[d1[:title], d2[:title]]
        if maxsim < s
          maxsim = s
          maxsim_pair = [d1, d2]
        end
      end
    end

    d1, d2 = maxsim_pair
    p [d1[:title], d2[:title], maxsim] if DEBUG
    merged << [d1[:title], d2[:title], maxsim]
    docs.delete(d2)
    docs.each do |d|
      next if d==d1
      v = sim[d[:title], d2[:title]]
      sim[d1[:title], d[:title]] = v if sim[d1[:title], d[:title]] < v
    end
  end
  merged
end


def pq_hac(docs)
  # similarity & priority queue
  sim = SymmetricHash.new
  clusters = Hash.new
  pqueue = Hash.new
  docs.each do |d1|
    t1 = d1[:title]
    clusters[t1] = [d1]
    pqueue[t1] = PriorityQueue.new
    docs.each do |d2|
      next if d1 == d2
      v = sim[t1, d2[:title]]
      unless v
        sim[t1, d2[:title]] = v = calc_sim(d1[:vector], d2[:vector])
        p [d1[:title], d2[:title], v] if DEBUG
      end
      pqueue[t1].insert( [v, d1, d2] )
    end
  end

  puts "---" if DEBUG

  # hac
  merged = []
  while docs.size > 1
    maxsim = [0]
    docs.each do |d1|
      t1 = d1[:title]
      maxsim = pqueue[t1].max if pqueue[t1].max[0] > maxsim[0]
    end
    v, d1, d2 = maxsim
    p [d1[:title], d2[:title], v] if DEBUG
    t1, t2 = d1[:title], d2[:title]
    merged << [t1, t2, v]
    docs.delete(d2)
    clusters[t1] += clusters[t2]
    pqueue[d1[:title]].clear
    docs.each do |d|
      next if d==d1
      t = d[:title]
      pqueue[t].delete_if_3rd(d1)
      pqueue[t].delete_if_3rd(d2)
      v = yield sim, clusters, d, d1, d2
      puts "  #{[d1[:title], t, v].inspect}" if DEBUG
      sim[d[:title], d1[:title]] = v
      pqueue[t].insert( [v, d, d1] )
      pqueue[t1].insert( [v, d1, d] )
    end
  end
  merged
end


complete_link = Proc.new do |sim, clusters, d, d1, d2|
  t, t1, t2 = d[:title], d1[:title], d2[:title]
  v = sim[t, t1]
  v = sim[t, t2] if v > sim[t, t2]
  v
end

single_link = Proc.new do |sim, clusters, d, d1, d2|
  t, t1, t2 = d[:title], d1[:title], d2[:title]
  v = sim[t, t1]
  v = sim[t, t2] if v < sim[t, t2]
  v
end

centroid = Proc.new do |sim, clusters, d, d1, d2|
  calc_sim(calc_centroid(clusters[d1[:title]]), calc_centroid(clusters[d[:title]]))
end
group_average = Proc.new do |sim, clusters, d, d1, d2|
  v = sum_vectors(clusters[d1[:title]] + clusters[d[:title]])
  n = clusters[d1[:title]].size + clusters[d[:title]].size
  (calc_sim(v, v) - n)/(n*(n-1))
end

def sum_vectors(docs)
  docs.map{|v| v[:vector]}.inject{|v1, v2| i=-1;v1.map{|x| x+v2[i+=1]}}
end
def calc_centroid(docs)
  sum_vectors(docs).map{|x| x/vectors.size}
end


clusters = pq_hac(docs, &group_average)



tree = Dendrogram.new(clusters)
clusters.each do |d1, d2, s|
  tree.branch d1, d2
end
tree.gen_output clusters
puts tree


