// Copyright 2026 The Net Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"archive/zip"
	"bytes"
	"embed"
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"math/rand"
	"sort"
	"strconv"
)

//go:embed iris.zip
var Iris embed.FS

// Fisher is the fisher iris data
type Fisher struct {
	Measures []float64
	Count    float64
	Embed    [10]float64
	Label    string
	Index    int
	Links    []float64
}

// Labels maps iris labels to ints
var Labels = map[string]int{
	"Iris-setosa":     0,
	"Iris-versicolor": 1,
	"Iris-virginica":  2,
}

// Inverse is the labels inverse map
var Inverse = [4]string{
	"Iris-setosa",
	"Iris-versicolor",
	"Iris-virginica",
}

// Load loads the iris data set
func Load() []Fisher {
	file, err := Iris.Open("iris.zip")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		panic(err)
	}

	fisher := make([]Fisher, 0, 8)
	reader, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		panic(err)
	}
	for _, f := range reader.File {
		if f.Name == "iris.data" {
			iris, err := f.Open()
			if err != nil {
				panic(err)
			}
			reader := csv.NewReader(iris)
			data, err := reader.ReadAll()
			if err != nil {
				panic(err)
			}
			for i, item := range data {
				record := Fisher{
					Measures: make([]float64, 4),
					Label:    item[4],
					Index:    i,
				}
				for ii := range item[:4] {
					f, err := strconv.ParseFloat(item[ii], 64)
					if err != nil {
						panic(err)
					}
					record.Measures[ii] = f
				}
				fisher = append(fisher, record)
			}
			iris.Close()
		}
	}
	return fisher
}

// Dot is the dot product
func Dot(a, b []float64) float64 {
	sum := 0.0
	for i, value := range a {
		sum += value * b[i]
	}
	return sum
}

// CS implements cosine similarity
func CS(a, b []float64) float64 {
	ab := Dot(a, b)
	aa := Dot(a, a)
	bb := Dot(b, b)
	if aa == 0 || bb == 0 {
		return 0
	}
	return ab / (math.Sqrt(aa) * math.Sqrt(bb))
}

func main() {
	iris := Load()

	rng := rand.New(rand.NewSource(1))
	samples := make([]float64, 0, 8)
	for range 33 {
		sum := 0.0
		for range 150 {
			sum += rng.Float64()
		}
		samples = append(samples, sum/150.0)
	}
	sort.Slice(samples, func(i, j int) bool {
		return samples[i] < samples[j]
	})
	for _, sample := range samples {
		fmt.Println(sample)
	}

	s := make([][]float64, 10)
	for i := range s {
		for range 150 {
			s[i] = append(s[i], rng.Float64())
		}
	}

	count := 0.0
	counts := make([]float64, 10)
	for i := range iris {
		for j := range iris {
			iris[i].Links = append(iris[i].Links, math.Abs(CS(iris[i].Measures, iris[j].Measures)))
		}
		fmt.Println(iris[i].Links)
	}
	index := 0
	for range 1024 * 1024 {
		/*samples := make([]float64, len(iris[index].Links))
		for i := range samples {
			samples[i] = rng.Float64()
		}*/
		ss := index % 10
		samples := s[ss]
		sum := 0.0
		for _, label := range iris[index].Links {
			sum += label //+ samples[i]
		}
		_ = samples
		link, selected, total := 0, rng.Float64(), 0.0
		for i, label := range iris[index].Links {
			total += (label /*+ samples[i]*/) / sum
			if selected < total {
				link = i
				break
			}
		}
		iris[link].Count++
		count++
		//iris[link].Embed[int(10*samples[link])]++
		//counts[int(10*samples[link])]++
		iris[link].Embed[ss]++
		counts[ss]++
		index = link
	}

	for i := range iris {
		fmt.Printf("(%f) ", iris[i].Count/count)
		for j := range iris[i].Embed {
			fmt.Printf("%f ", iris[i].Embed[j]/counts[j])
		}
		fmt.Println()
	}
}
