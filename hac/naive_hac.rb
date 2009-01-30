#!/usr/bin/ruby

def calc_tdidf(tf, df, n_docs)
  tf * Math.log(n_docs / df)
end
def calc_sim(v1, v2)
  sim = 0
  v1.each_with_index {|x, i| sim += x * v2[i] }
  sim
end

class Similarity
  def initialize
    @memo = Hash.new
  end
  def calc(d1, d2)
    key = [d1[:title], d2[:title]]
    @memo[key] || @memo[key.reverse] || (@memo[key] = calc_sim(d1[:vector], d2[:vector]))
  end
  def include?(pair, title)
    return pair[0] if pair[1] == title
    return pair[1] if pair[0] == title
  end
  def merge(d1, d2)
    @memo.each do |key, value|
      if (title2 = include?(key, d1[:title])) && title2 != d2[:title]
        value2 = @memo[[title2, d2[:title]]] || @memo[[d2[:title], title2]]
        @memo[key] = value2 if value2 > value
      end
    end
  end
end


data = open(ARGV[0] || 'corpus'){|f| Marshal.load(f) }
docs = data[:docs]
terms = data[:terms]


# all terms
#coordinates = terms.keys

# 2`n_docs-1
coordinates = terms.keys.select{|term| n=terms[term].size; n>1 && n<docs.size }
#puts coordinates.size


docs.each_with_index do |doc, doc_id|
  l2 = 0
  v = coordinates.map do |term|
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


sim = Similarity.new
clusters = []
while docs.size > 1
  puts docs.size
  maxsim = 0
  maxsim_pair = nil
  docs.each do |d1|
    docs.each do |d2|
      break if d1==d2
      s = sim.calc(d1, d2)
      if maxsim < s
        maxsim = s
        maxsim_pair = [d1, d2]
      end
    end
  end

  clusters << [maxsim_pair[0][:title], maxsim_pair[1][:title], maxsim]
  docs.delete(maxsim_pair[1])
  sim.merge maxsim_pair[0], maxsim_pair[1]
end

p clusters


class Tree
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

tree = Tree.new(clusters)
clusters.each do |d1, d2, s|
  tree.branch d1, d2
end
tree.gen_output clusters
puts tree


