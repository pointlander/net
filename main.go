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

	"github.com/pointlander/net/kmeans"
)

//go:embed iris.zip
var Iris embed.FS

// Fisher is the fisher iris data
type Fisher struct {
	Measures []float64
	Count    float64
	Embed    [150]float64
	Label    string
	Index    int
	Links    []float64
	Cluster  int
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

	count := 0.0
	counts := make([]float64, 150)
	for i := range iris {
		for j := range iris {
			iris[i].Links = append(iris[i].Links, math.Abs(CS(iris[i].Measures, iris[j].Measures)))
		}
		fmt.Println(iris[i].Links)
	}
	index := 0
	for range 8 * 1024 * 1024 {
		samples := make([]float64, len(iris[index].Links))
		for i := range samples {
			samples[i] = rng.Float64()
		}
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
		type Sample struct {
			Sample float64
			Index  int
		}
		s := make([]Sample, len(iris[index].Links))
		for i, label := range iris[index].Links {
			s[i].Sample = label
			s[i].Index = i
		}
		sort.Slice(s, func(i, j int) bool {
			return s[i].Sample < s[j].Sample
		})
		l := 0
		for i, value := range s {
			if value.Index == link {
				l = i
				break
			}
		}
		iris[link].Embed[l]++
		counts[l]++
		index = link
	}

	meta := make([][]float64, len(iris))
	for i := range meta {
		meta[i] = make([]float64, len(iris))
	}
	const k = 3

	{
		vectors := make([][]float64, len(iris))
		for i := range vectors {
			vector := make([]float64, len(iris))
			for ii := range iris[i].Embed {
				vector[ii] = iris[i].Embed[ii] / counts[ii]
			}
			vectors[i] = vector
		}
		for i := 0; i < 33; i++ {
			clusters, _, err := kmeans.Kmeans(int64(i+1), vectors, k, kmeans.SquaredEuclideanDistance, -1)
			if err != nil {
				panic(err)
			}
			for i := 0; i < len(meta); i++ {
				target := clusters[i]
				for j, v := range clusters {
					if v == target {
						meta[i][j]++
					}
				}
			}
		}
	}
	clusters, _, err := kmeans.Kmeans(1, meta, 3, kmeans.SquaredEuclideanDistance, -1)
	if err != nil {
		panic(err)
	}
	for i := range clusters {
		iris[i].Cluster = clusters[i]
	}

	sort.Slice(iris, func(i, j int) bool {
		return iris[i].Cluster < iris[j].Cluster
	})

	for i := range iris {
		fmt.Printf("%s (%f) ", iris[i].Label, iris[i].Count/count)
		/*for j := range iris[i].Embed {
			fmt.Printf("%f ", iris[i].Embed[j]/counts[j])
		}*/
		fmt.Println()
	}
}
