#!/usr/bin/ruby

require 'rubygems'
require 'hpricot'

filename = ARGV[0]
text = open(filename){|f| f.read}
doc = Hpricot(text)

class Hmm
  def initialize
    @symbols = Hash.new
    @statuses = Hash.new
    @previous_symbol = nil
    @current_status = nil
  end
  def transrate(symbol)
    if @previous_symbol
      @symbols[symbol] = :dummy
    end
    @previous_symbol = symbol
  end
  def walk(node)
    if node.text?
      text = node.inner_text.strip.gsub(/\n/, '\n')
      return if text == ""
      transrate :text
    elsif node.comment?
      if node.to_s =~ /google_ad_section_start/
        @current_status = :body
      elsif node.to_s =~ /google_ad_section_end/
        @current_status = nil
      end
      #symbol :comment
    elsif node.elem?
      if node.children==0
      else
        puts "<#{node.name}>"
        node.children.each {|child| walk child }
      end
    else
      puts node.class
    end
    return if node.doctype? || node.bogusetag?
    return unless node.children
  end
end

hmm = Hmm.new
hmm.walk doc.at("body")




=begin
text.scan(/([^<]*)(<[^>]*>)/) do |x|
  s = x[0].strip
  puts s if s != ""
  puts x[1]
end
=end

