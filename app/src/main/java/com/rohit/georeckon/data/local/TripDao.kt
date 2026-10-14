package com.rohit.georeckon.data.local

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query

@Dao
interface TripDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertTrip(trip: TripEntity): Long

    @Query("SELECT * FROM trips ORDER BY startTimeMs DESC")
    suspend fun getAllTrips(): List<TripEntity>

    @Query("DELETE FROM trips")
    suspend fun deleteAllTrips()
}
